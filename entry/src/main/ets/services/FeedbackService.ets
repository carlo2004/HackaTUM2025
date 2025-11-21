// src/main/ets/services/FeedbackService.ts
import { vibrator } from '@kit.SensorServiceKit';

export class FeedbackService {

  // The Metronome Tick (Short, Sharp)
  triggerMetronome() {
    try {
      vibrator.startVibration({
        type: 'preset',
        effectId: 'haptic.clock.timer',
        count: 1,
      }, { usage: 'physicalFeedback' });
    } catch (e) {}
  }

  // The Warning Buzz (Long, Scary)
  triggerWarning() {
    try {
      vibrator.startVibration({
        type: 'time',
        duration: 500, // 0.5 seconds long buzz
      }, { usage: 'alarm' });
    } catch (e) {}
  }
}