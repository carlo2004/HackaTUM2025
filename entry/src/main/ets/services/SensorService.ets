// src/main/ets/services/SensorService.ts
import { sensor } from '@kit.SensorServiceKit';

export class SensorService {
  private isListening: boolean = false;
  private onCompressionDetected: (force: number) => void = () => {};

  constructor() {}

  // The Manager gives us a function to call when a push happens
  setCallback(callback: (force: number) => void) {
    this.onCompressionDetected = callback;
  }

  start() {
    if (this.isListening) return;
    this.isListening = true;

    try {
      // Subscribe to Accelerometer at GAME speed (20ms)
      sensor.on(sensor.SensorId.ACCELEROMETER, (data) => {
        // Raw Physics Filter
        if (data.z < -15) {
          this.onCompressionDetected(data.z);
        }
      }, { interval: 20000000 }); // 20ms
    } catch (e) {
      console.error("Sensor Error: " + e);
    }
  }

  stop() {
    this.isListening = false;
    sensor.off(sensor.SensorId.ACCELEROMETER);
  }
}