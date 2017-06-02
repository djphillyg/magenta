# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Performance RNN generation code as a SequenceGenerator interface."""

import copy
import math
from functools import partial

# internal imports

import tensorflow as tf

from magenta.models.performance_rnn import performance_lib
from magenta.models.performance_rnn import performance_model
from magenta.models.performance_rnn.performance_lib import PerformanceEvent

import magenta.music as mm


class PerformanceRnnSequenceGenerator(mm.BaseSequenceGenerator):
  """Performance RNN generation code as a SequenceGenerator interface."""

  def __init__(self, model, details,
               steps_per_second=performance_lib.DEFAULT_STEPS_PER_SECOND,
               checkpoint=None, bundle=None):
    """Creates a PerformanceRnnSequenceGenerator.

    Args:
      model: Instance of PolyphonyRnnModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_second: Number of quantized steps per second.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(PerformanceRnnSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self.steps_per_second = steps_per_second

  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
      raise mm.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise mm.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.trim_note_sequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = mm.quantize_to_step(
          input_section.start_time, self.steps_per_second)
    else:
      primer_sequence = input_sequence
      input_start_step = 0

    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    if last_end_time > generate_section.start_time:
      raise mm.SequenceGeneratorException(
          'Got GenerateSection request for section that is before or equal to '
          'the end of the NoteSequence. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time, last_end_time))

    # Quantize the priming sequence.
    quantized_primer_sequence = mm.quantize_note_sequence_absolute(
        primer_sequence, self.steps_per_second)

    extracted_perfs, _ = performance_lib.extract_performances(
        quantized_primer_sequence, start_step=input_start_step)
    assert len(extracted_perfs) <= 1

    generate_start_step = mm.quantize_to_step(
        generate_section.start_time, self.steps_per_second)
    # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
    # always rounds down. This avoids generating a sequence that ends at 5.0
    # seconds when the requested end time is 4.99.
    generate_end_step = mm.quantize_to_step(
        generate_section.end_time, self.steps_per_second, quantize_cutoff=1.0)

    if extracted_perfs and extracted_perfs[0]:
      performance = extracted_perfs[0]
    else:
      # If no track could be extracted, create an empty track that starts at the
      # requested generate_start_step.
      performance = performance_lib.Performance(
          steps_per_second=(
              quantized_primer_sequence.quantization_info.steps_per_second),
          start_step=generate_start_step)

    # Ensure that the track extends up to the step we want to start generating.
    performance.set_length(generate_start_step - performance.start_step)

    # Extract generation arguments from generator options.
    arg_types = {
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
    }
    args = dict((name, value_fn(generator_options.args[name]))
                for name, value_fn in arg_types.items()
                if name in generator_options.args)

    total_steps = performance.num_steps + (
        generate_end_step - generate_start_step)

    if not performance:
      # Primer is empty; let's just start with silence.
      performance.set_length(min(performance_lib.MAX_SHIFT_STEPS, total_steps))

    while performance.num_steps < total_steps:
      # Assume there's around 5 notes per second and 3 RNN steps per note. Can't
      # know for sure until generation is finished because the number of notes
      # per quantized step is variable.
      steps_to_gen = total_steps - performance.num_steps
      rnn_steps_to_gen = int(math.ceil(
          15.0 * steps_to_gen / performance_lib.DEFAULT_STEPS_PER_SECOND))
      tf.logging.info(
          'Need to generate %d more steps for this sequence, will try asking '
          'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
      performance = self._model.generate_performance(
          len(performance) + rnn_steps_to_gen, performance, **args)
    performance.set_length(total_steps)

    generated_sequence = performance.to_sequence()

    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence


def get_generator_map():
  """Returns a map from the generator ID to a SequenceGenerator class creator.

  Binds the `config` argument so that the arguments match the
  BaseSequenceGenerator class constructor.

  Returns:
    Map from the generator ID to its SequenceGenerator class creator with a
    bound `config` argument.
  """
  def create_sequence_generator(config, **kwargs):
    return PerformanceRnnSequenceGenerator(
        performance_model.PerformanceRnnModel(config), config.details,
        steps_per_second=config.steps_per_second, **kwargs)

  return {key: partial(create_sequence_generator, config)
          for (key, config) in performance_model.default_configs.items()}
