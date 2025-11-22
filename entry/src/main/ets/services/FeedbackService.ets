import { vibrator } from '@kit.SensorServiceKit';

export class FeedbackService {

  triggerMetronome() {
    try {
      vibrator.startVibration({
        type: 'preset',
        effectId: 'haptic.clock.timer',
        count: 1,
      }, { usage: 'physicalFeedback' });
    } catch (e) {}
  }

  triggerError() {
    try {
      vibrator.startVibration({
        type: 'time',
        duration: 800,
      }, { usage: 'alarm' });
    } catch (e) {}
  }
}