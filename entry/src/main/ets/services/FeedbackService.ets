import { vibrator } from '@kit.SensorServiceKit';

export class FeedbackService {

  triggerMetronome() {
    try {

      vibrator.startVibration({
        type: 'time',
        duration: 100,
      }, { usage: 'alarm' });
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