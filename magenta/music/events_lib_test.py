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
"""Tests for events_lib."""

import copy

# internal imports
import tensorflow as tf

from magenta.music import events_lib


class EventSequenceTest(tf.test.TestCase):

  def testDeepcopy(self):
    events = events_lib.SimpleEventSequence(
        pad_event=0, events=[0, 1, 2], start_step=0, steps_per_quarter=4,
        steps_per_bar=8)
    events_copy = copy.deepcopy(events)
    self.assertEqual(events, events_copy)

    events.set_length(2)
    self.assertNotEqual(events, events_copy)

  def testAppend(self):
    events = events_lib.SimpleEventSequence(pad_event=0)

    events.append(7)
    self.assertListEqual([7], list(events))
    self.assertEqual(0, events.start_step)
    self.assertEqual(1, events.end_step)

    events.append('cheese')
    self.assertListEqual([7, 'cheese'], list(events))
    self.assertEqual(0, events.start_step)
    self.assertEqual(2, events.end_step)

  def testSetLength(self):
    events = events_lib.SimpleEventSequence(
        pad_event=0, events=[60], start_step=9)
    events.set_length(5)
    self.assertListEqual([60, 0, 0, 0, 0],
                         list(events))
    self.assertEquals(9, events.start_step)
    self.assertEquals(14, events.end_step)

    events = events_lib.SimpleEventSequence(
        pad_event=0, events=[60], start_step=9)
    events.set_length(5, from_left=True)
    self.assertListEqual([0, 0, 0, 0, 60],
                         list(events))
    self.assertEquals(5, events.start_step)
    self.assertEquals(10, events.end_step)

    events = events_lib.SimpleEventSequence(pad_event=0, events=[60, 0, 0, 0])
    events.set_length(3)
    self.assertListEqual([60, 0, 0], list(events))
    self.assertEquals(0, events.start_step)
    self.assertEquals(3, events.end_step)

    events = events_lib.SimpleEventSequence(pad_event=0, events=[60, 0, 0, 0])
    events.set_length(3, from_left=True)
    self.assertListEqual([0, 0, 0], list(events))
    self.assertEquals(1, events.start_step)
    self.assertEquals(4, events.end_step)

  def testIncreaseResolution(self):
    events = events_lib.SimpleEventSequence(pad_event=0, events=[1, 0, 1, 0],
                                            start_step=5, steps_per_bar=4,
                                            steps_per_quarter=1)
    events.increase_resolution(3, fill_event=None)
    self.assertListEqual([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], list(events))
    self.assertEquals(events.start_step, 15)
    self.assertEquals(events.steps_per_bar, 12)
    self.assertEquals(events.steps_per_quarter, 3)

    events = events_lib.SimpleEventSequence(pad_event=0, events=[1, 0, 1, 0])
    events.increase_resolution(2, fill_event=0)
    self.assertListEqual([1, 0, 0, 0, 1, 0, 0, 0], list(events))


class RunLengthEncodedEventSequenceTest(tf.test.TestCase):

  def testEncode(self):
    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=['A', 'A', 'A', 'B', 'C', 'C'], start_step=0)
    events = events_lib.RunLengthEncodedEventSequence(base_events, 2)
    expected_events = [('A', 2), ('A', 1), ('B', 1), ('C', 2)]
    self.assertListEqual(expected_events, list(events))
    self.assertEqual(base_events.start_step, events.start_step)
    self.assertEqual(base_events.end_step, events.end_step)

  def testEncodeOnlyEncodePadEvent(self):
    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=['A', 'A', '_', '_', '_', 'B'], start_step=0)
    events = events_lib.RunLengthEncodedEventSequence(
        base_events, 2, only_encode_pad_event=True)
    expected_events = [('A', 1), ('A', 1), ('_', 2), ('_', 1), ('B', 1)]
    self.assertListEqual(expected_events, list(events))
    self.assertEqual(base_events.start_step, events.start_step)
    self.assertEqual(base_events.end_step, events.end_step)

  def testAppend(self):
    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=[], start_step=0)
    events = events_lib.RunLengthEncodedEventSequence(base_events, 2)

    events.append(('A', 2))
    self.assertListEqual([('A', 2)], list(events))
    self.assertEqual(0, events.start_step)
    self.assertEqual(2, events.end_step)

    events.append(('B', 1))
    self.assertListEqual([('A', 2), ('B', 1)], list(events))
    self.assertEqual(0, events.start_step)
    self.assertEqual(3, events.end_step)

  def testSetLength(self):
    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=[], start_step=1)
    events = events_lib.RunLengthEncodedEventSequence(base_events, 2)

    events.set_length(3)
    self.assertListEqual([('_', 2), ('_', 1)], list(events))
    self.assertEqual(1, events.start_step)
    self.assertEqual(4, events.end_step)

    events.set_length(4, from_left=True)
    self.assertListEqual([('_', 2), ('_', 2)], list(events))
    self.assertEqual(0, events.start_step)
    self.assertEqual(4, events.end_step)

    events.set_length(2)
    self.assertListEqual([('_', 2)], list(events))
    self.assertEqual(0, events.start_step)
    self.assertEqual(2, events.end_step)

    events.set_length(1, from_left=True)
    self.assertListEqual([('_', 1)], list(events))
    self.assertEqual(1, events.start_step)
    self.assertEqual(2, events.end_step)

    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=['A', 'A'], start_step=10)
    events = events_lib.RunLengthEncodedEventSequence(base_events, 2)

    events.set_length(5)
    self.assertListEqual([('A', 2), ('_', 2), ('_', 1)], list(events))
    self.assertEqual(10, events.start_step)
    self.assertEqual(15, events.end_step)

    events.set_length(6, from_left=True)
    self.assertListEqual([('_', 1), ('A', 2), ('_', 2), ('_', 1)], list(events))
    self.assertEqual(9, events.start_step)
    self.assertEqual(15, events.end_step)

    events.set_length(2)
    self.assertListEqual([('_', 1), ('A', 1)], list(events))
    self.assertEqual(9, events.start_step)
    self.assertEqual(11, events.end_step)

  def testSetLengthOnlyEncodePadEvent(self):
    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=['A', 'A'], start_step=10)
    events = events_lib.RunLengthEncodedEventSequence(
        base_events, 2, only_encode_pad_event=True)

    events.set_length(5)
    self.assertListEqual([('A', 1), ('A', 1), ('_', 2), ('_', 1)], list(events))
    self.assertEqual(10, events.start_step)
    self.assertEqual(15, events.end_step)

    events.set_length(6, from_left=True)
    self.assertListEqual([('_', 1), ('A', 1), ('A', 1), ('_', 2), ('_', 1)],
                         list(events))
    self.assertEqual(9, events.start_step)
    self.assertEqual(15, events.end_step)

    events.set_length(2)
    self.assertListEqual([('_', 1), ('A', 1)], list(events))
    self.assertEqual(9, events.start_step)
    self.assertEqual(11, events.end_step)

  def testDecode(self):
    base_events = events_lib.SimpleEventSequence(
        pad_event='_', events=['A', 'A', 'A', 'B', 'C', 'C'], start_step=0)
    events = events_lib.RunLengthEncodedEventSequence(base_events, 2)

    decoded_events = copy.deepcopy(base_events)
    decoded_events.set_length(0)
    events.decode(decoded_events)

    self.assertListEqual(list(base_events), list(decoded_events))
    self.assertEqual(base_events.start_step, decoded_events.start_step)
    self.assertEqual(base_events.end_step, decoded_events.end_step)


if __name__ == '__main__':
  tf.test.main()
