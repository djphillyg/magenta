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
"""Tests for performance_encoder_decoder."""

# internal imports

import tensorflow as tf

from magenta.models.performance_rnn import performance_encoder_decoder
from magenta.models.performance_rnn.performance_lib import PerformanceEvent


class PerformanceOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = performance_encoder_decoder.PerformanceOneHotEncoding(
        num_velocity_bins=16)

  def testEncodeDecode(self):
    note_on_1 = PerformanceEvent(
        event_type=PerformanceEvent.NOTE_ON, event_value=60)
    note_on_2 = PerformanceEvent(
        event_type=PerformanceEvent.NOTE_ON, event_value=0)
    note_on_3 = PerformanceEvent(
        event_type=PerformanceEvent.NOTE_ON, event_value=127)
    note_off_1 = PerformanceEvent(
        event_type=PerformanceEvent.NOTE_OFF, event_value=72)
    note_off_2 = PerformanceEvent(
        event_type=PerformanceEvent.NOTE_OFF, event_value=0)
    note_off_3 = PerformanceEvent(
        event_type=PerformanceEvent.NOTE_OFF, event_value=127)
    time_shift_1 = PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT, event_value=10)
    time_shift_2 = PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT, event_value=1)
    time_shift_3 = PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT, event_value=100)
    velocity_1 = PerformanceEvent(
        event_type=PerformanceEvent.VELOCITY, event_value=5)
    velocity_2 = PerformanceEvent(
        event_type=PerformanceEvent.VELOCITY, event_value=1)
    velocity_3 = PerformanceEvent(
        event_type=PerformanceEvent.VELOCITY, event_value=16)

    index = self.enc.encode_event(note_on_1)
    self.assertEqual(60, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, note_on_1)
    index = self.enc.encode_event(note_on_2)
    self.assertEqual(0, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, note_on_2)
    index = self.enc.encode_event(note_on_3)
    self.assertEqual(127, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, note_on_3)

    index = self.enc.encode_event(note_off_1)
    self.assertEqual(200, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, note_off_1)
    index = self.enc.encode_event(note_off_2)
    self.assertEqual(128, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, note_off_2)
    index = self.enc.encode_event(note_off_3)
    self.assertEqual(255, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, note_off_3)

    index = self.enc.encode_event(time_shift_1)
    self.assertEqual(265, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, time_shift_1)
    index = self.enc.encode_event(time_shift_2)
    self.assertEqual(256, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, time_shift_2)
    index = self.enc.encode_event(time_shift_3)
    self.assertEqual(355, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, time_shift_3)

    index = self.enc.encode_event(velocity_1)
    self.assertEqual(360, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, velocity_1)
    index = self.enc.encode_event(velocity_2)
    self.assertEqual(356, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, velocity_2)
    index = self.enc.encode_event(velocity_3)
    self.assertEqual(371, index)
    event = self.enc.decode_event(index)
    self.assertEqual(event, velocity_3)


if __name__ == '__main__':
  tf.test.main()
