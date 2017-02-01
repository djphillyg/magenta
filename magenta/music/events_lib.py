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
"""Abstract base classes for working with musical event sequences.

The abstract `EventSequence` class is an interface for a sequence of musical
events. The `SimpleEventSequence` class is a basic implementation of this
interface.
"""

import abc
import copy

# internal imports
from magenta.music import constants


DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER
STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_MAX_RUN_LENGTH = DEFAULT_STEPS_PER_BAR


class NonIntegerStepsPerBarException(Exception):
  pass


class RunLengthEncodedEventSequenceException(Exception):
  pass


class EventSequence(object):
  """Stores a quantized stream of events.

  EventSequence is an abstract class to use as an interface for interacting
  with musical event sequences. Concrete implementations SimpleEventSequence
  (and its descendants Melody and ChordProgression) and LeadSheet represent
  sequences of musical events of particular types. In all cases, model-specific
  code is responsible for converting this representation to SequenceExample
  protos for TensorFlow.

  EventSequence represents an iterable object. Simply iterate to retrieve the
  events.

  Attributes:
    start_step: The offset of the first step of the sequence relative to the
        beginning of the source sequence. Should always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
        of the sequence relative to the beginning of the source sequence. Will
        always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def start_step(self):
    pass

  @abc.abstractproperty
  def end_step(self):
    pass

  @abc.abstractproperty
  def steps_per_quarter(self):
    pass

  @abc.abstractmethod
  def append(self, event):
    """Appends event to the end of the sequence.

    Args:
      event: The event to append to the end.
    """
    pass

  @abc.abstractmethod
  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, will pad to make the sequence
    the specified length. If it is too long, it will be truncated to the
    requested length.

    Args:
      steps: How many steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    pass

  @abc.abstractmethod
  def __getitem__(self, i):
    """Returns the event at the given index."""
    pass

  @abc.abstractmethod
  def __iter__(self):
    """Returns an iterator over the events."""
    pass

  @abc.abstractmethod
  def __len__(self):
    """How many events are in this EventSequence.

    Returns:
      Number of events as an integer.
    """
    pass


class SimpleEventSequence(EventSequence):
  """Stores a quantized stream of events.

  This class can be instantiated, but its main purpose is to serve as a base
  class for Melody, ChordProgression, and any other simple stream of musical
  events.

  SimpleEventSequence represents an iterable object. Simply iterate to retrieve
  the events.

  Attributes:
    pad_event: Event value to use when padding sequences.
    start_step: The offset of the first step of the sequence relative to the
        beginning of the source sequence. Should always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
       of the sequence relative to the beginning of the source sequence. Will
       always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
    steps_per_bar: Number of steps in a bar (measure) of music.
  """

  def __init__(self, pad_event, events=None, start_step=0,
               steps_per_bar=DEFAULT_STEPS_PER_BAR,
               steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Construct a SimpleEventSequence.

    If `events` is specified, instantiate with the provided event list.
    Otherwise, create an empty SimpleEventSequence.

    Args:
      pad_event: Event value to use when padding sequences.
      events: List of events to instantiate with.
      start_step: The integer starting step offset.
      steps_per_bar: The number of steps in a bar.
      steps_per_quarter: The number of steps in a quarter note.
    """
    self._pad_event = pad_event
    if events is not None:
      self._from_event_list(events, start_step=start_step,
                            steps_per_bar=steps_per_bar,
                            steps_per_quarter=steps_per_quarter)
    else:
      self._events = []
      self._steps_per_bar = steps_per_bar
      self._steps_per_quarter = steps_per_quarter
      self._start_step = start_step
      self._end_step = start_step

  def _reset(self):
    """Clear events and reset object state."""
    self._events = []
    self._steps_per_bar = DEFAULT_STEPS_PER_BAR
    self._steps_per_quarter = DEFAULT_STEPS_PER_QUARTER
    self._start_step = 0
    self._end_step = 0

  def _from_event_list(self, events, start_step=0,
                       steps_per_bar=DEFAULT_STEPS_PER_BAR,
                       steps_per_quarter=DEFAULT_STEPS_PER_QUARTER):
    """Initializes with a list of event values and sets attributes."""
    self._events = list(events)
    self._start_step = start_step
    self._end_step = start_step + len(self)
    self._steps_per_bar = steps_per_bar
    self._steps_per_quarter = steps_per_quarter

  def __iter__(self):
    """Return an iterator over the events in this SimpleEventSequence.

    Returns:
      Python iterator over events.
    """
    return iter(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __getslice__(self, i, j):
    """Returns the events in the given slice range."""
    return self._events[i:j]

  def __len__(self):
    """How many events are in this SimpleEventSequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __deepcopy__(self, unused_memo=None):
    return type(self)(pad_event=self.pad_event,
                      events=copy.deepcopy(self._events),
                      start_step=self.start_step,
                      steps_per_bar=self.steps_per_bar,
                      steps_per_quarter=self.steps_per_quarter)

  def __eq__(self, other):
    if not isinstance(other, SimpleEventSequence):
      return False
    return (list(self) == list(other) and
            self.steps_per_bar == other.steps_per_bar and
            self.steps_per_quarter == other.steps_per_quarter and
            self.start_step == other.start_step and
            self.end_step == other.end_step)

  @property
  def pad_event(self):
    return self._pad_event

  @property
  def start_step(self):
    return self._start_step

  @property
  def end_step(self):
    return self._end_step

  @property
  def steps_per_bar(self):
    return self._steps_per_bar

  @property
  def steps_per_quarter(self):
    return self._steps_per_quarter

  def append(self, event):
    """Appends event to the end of the sequence and increments the end step.

    Args:
      event: The event to append to the end.
    """
    self._events.append(event)
    self._end_step += 1

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads to make the sequence the
    specified length. If it is too long, it will be truncated to the requested
    length.

    Args:
      steps: How many steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if steps > len(self):
      if from_left:
        self._events[:0] = [self._pad_event] * (steps - len(self))
      else:
        self._events.extend([self._pad_event] * (steps - len(self)))
    else:
      if from_left:
        del self._events[0:-steps]
      else:
        del self._events[steps:]

    if from_left:
      self._start_step = self._end_step - steps
    else:
      self._end_step = self._start_step + steps

  def increase_resolution(self, k, fill_event=None):
    """Increase the resolution of an event sequence.

    Increases the resolution of a SimpleEventSequence object by a factor of
    `k`.

    Args:
      k: An integer, the factor by which to increase the resolution of the
          event sequence.
      fill_event: Event value to use to extend each low-resolution event. If
          None, each low-resolution event value will be repeated `k` times.
    """
    if fill_event is None:
      fill = lambda event: [event] * k
    else:
      fill = lambda event: [event] + [fill_event] * (k - 1)

    new_events = []
    for event in self._events:
      new_events += fill(event)

    self._events = new_events
    self._start_step *= k
    self._end_step *= k
    self._steps_per_bar *= k
    self._steps_per_quarter *= k


class RunLengthEncodedEventSequence(EventSequence):
  """Stores a run-length encoded sequence of events.

  Each "event" in the run-length encoded sequence is a tuple consisting of base
  event and run-length (count), where run-length is an integer between 1 and
  `max_run_length`.

  If the underlying event sequence contains more than `max_run_length`
  occurrences in a row of the same base event, this will be represented as
  multiple events in the run-length encoded sequence.

  Attributes:
    start_step: The offset of the first step of the sequence relative to the
        beginning of the source sequence. Should always be the first step of a
        bar.
    end_step: The offset to the beginning of the bar following the last step
       of the sequence relative to the beginning of the source sequence. Will
       always be the first step of a bar.
    steps_per_quarter: Number of steps in in a quarter note.
  """

  def __init__(self, base_events, max_run_length=DEFAULT_MAX_RUN_LENGTH):
    """Construct a RunLengthEncodedEventSequence from a SimpleEventSequence.

    This applies run-length encoding to the base sequence, compressing repeated
    events into (event, run-length) tuples.

    Args:
      base_events: A SimpleEventSequence to run-length encode.
      max_run_length: The maximum run-length to use. Base events that repeat
          consecutively more times than this will be split into multiple events
          in the run-length encoded sequence.

    Raises:
      RunLengthEncodedEventSequenceException: If `base_events` is not an object
          of type SimpleEventSequence.
    """
    if not isinstance(base_events, SimpleEventSequence):
      raise RunLengthEncodedEventSequenceException(
          'base event sequence must have type SimpleEventSequence')

    self._pad_event = base_events.pad_event
    self._max_run_length = max_run_length
    self._events = []
    self._start_step = base_events.start_step
    self._steps_per_quarter = base_events.steps_per_quarter

    current_event = None
    current_run_length = 0

    for event in base_events:
      if event != current_event or current_run_length == max_run_length:
        if current_run_length > 0:
          self._events.append((current_event, current_run_length))
        current_event = event
        current_run_length = 0
      current_run_length += 1

    if current_run_length > 0:
      self._events.append((current_event, current_run_length))

  @property
  def num_steps(self):
    """Return the number of steps in the underlying event sequence.

    Returns:
      The number of steps in the underlying event sequence. (This is always at
      least as large as the number of events in the run-length encoding.)
    """
    return sum(run_length for event, run_length in self._events)

  @property
  def start_step(self):
    return self._start_step

  @property
  def end_step(self):
    return self.start_step + self.num_steps

  @property
  def steps_per_quarter(self):
    return self._steps_per_quarter

  def append(self, event):
    """Appends a run-length encoded event to the end of the sequence.

    Args:
      event: The event (a tuple of base event and run-length) to append.

    Raises:
      ValueError: If `event` is not a length-2 tuple where the 2nd element is an
          integer between 1 and `self.max_run_length`.
    """
    if not isinstance(event, tuple) or len(event) != 2:
      raise ValueError(
          'run-length encoded event must be (base event, run-length) tuple')
    base_event, run_length = event
    if not isinstance(run_length, int) or run_length < 1:
      raise ValueError('run-length must be positive integer')
    if run_length > self._max_run_length:
      raise ValueError('run-length exceeds maximum run-length')

    self._events.append((base_event, run_length))

  def _padding(self, steps):
    """Generate run-length encoded padding.

    Args:
      steps: The number of (underlying) steps of padding to generate.

    Returns:
      A Python list of run-length encoded padding (`self._pad_event`), consuming
      `steps` steps.
    """
    events = []
    while steps > 0:
      added_steps = min(steps, self._max_run_length)
      events.append((self._pad_event, added_steps))
      steps -= added_steps
    return events

  def _standardize_run_lengths(self):
    """Standardize the run-lengths of this RunLengthEncodedEventSequence.

    Run-length encodings are not unique, e.g. [('A', 2), ('A', 3)] and
    [('A', 3), ('A', 2)] both decode to the same underlying sequence. This
    method standardizes run-lengths so that a run-length encoded event with run-
    length less than `self._max_run_length` cannot precede a run-length encoded
    event with the same underlying event.
    """
    current_event = None
    current_run_length = 0

    new_events = []
    for event, run_length in self._events:
      if event != current_event:
        if current_run_length > 0:
          new_events.append((current_event, current_run_length))
        current_event = event
        current_run_length = 0
      current_run_length += run_length
      if current_run_length >= self._max_run_length:
        new_events.append((current_event, self._max_run_length))
        current_run_length -= self._max_run_length

    if current_run_length > 0:
      new_events.append((current_event, current_run_length))

    self._events = new_events

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads to make the sequence the
    specified length. If it is too long, it will be truncated to the requested
    length.

    Args:
      steps: How many steps long the (underlying) event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if steps > self.num_steps:
      # Add padding to run-length encoded event sequence.
      steps_to_add = steps - self.num_steps
      if from_left:
        self._start_step -= steps_to_add
        self._events = self._padding(steps_to_add) + self._events
      else:
        self._events += self._padding(steps_to_add)

    else:
      # Remove events from run-length encoded sequence.
      steps_to_remove = self.num_steps - steps
      if from_left:
        self._start_step += steps_to_remove
        while steps_to_remove > 0:
          event, run_length = self._events[0]
          if run_length > steps_to_remove:
            self._events[0] = (event, run_length - steps_to_remove)
            steps_to_remove = 0
          else:
            self._events = self._events[0:]
            steps_to_remove -= run_length
      else:
        while steps_to_remove > 0:
          event, run_length = self._events[-1]
          if run_length > steps_to_remove:
            self._events[-1] = (event, run_length - steps_to_remove)
            steps_to_remove = 0
          else:
            self._events = self._events[:-1]
            steps_to_remove -= run_length

    # Fix the sequence so that there are no non-standard run-lengths, e.g. an
    # event with non-maximum run-length followed by the same event.
    self._standardize_run_lengths()

  def __getitem__(self, i):
    """Returns the run-length encoded event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Returns an iterator over run-length encoded events."""
    return iter(self._events)

  def __len__(self):
    """How many run-length encoded events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def decode(self, base_events):
    """Decode run-length encoded event sequence into base sequence.

    Args:
      base_events: The base SimpleEventSequence object into which to decode the
          run-length encoded sequence.

    Raises:
      RunLengthEncodedEventSequenceException: If `base_events` is non-empty.
    """
    if base_events:
      raise RunLengthEncodedEventSequenceException(
          'cannot decode into non-empty event sequence')

    for (event, run_length) in self._events:
      for _ in range(run_length):
        base_events.append(event)
