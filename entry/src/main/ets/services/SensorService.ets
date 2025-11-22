import { sensor } from '@kit.SensorServiceKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

export class SensorBundle {
  accelX: number = 0;
  accelY: number = 0;
  accelZ: number = 0;
  gyroX: number = 0;
  gyroY: number = 0;
  gyroZ: number = 0;
  heartRate: number = 0;
  oxygen: number = 0;
  temperature: number = 0;
}

export class SensorService {

  private SIMULATION_MODE: boolean = false;

  private accelX: number = 0;
  private accelY: number = 0;
  private accelZ: number = 0;

  private bufX: number[] = [0, 0, 0];
  private bufY: number[] = [0, 0, 0];
  private bufZ: number[] = [0, 0, 0];

  private isListening: boolean = false;

  private gravityEstimate: number = 9.8;
  private lastBeatTime: number = 0;

  private beatTimestamps: number[] = [];

  private inactivityTimer: number = 0;

  private WINDOW_SIZE_MS: number = 5000;

  private MIN_BEAT_INTERVAL: number = 350;

  private PUSH_THRESHOLD: number = 2.8;

  private onCompressionDetected: (bpm: number, depth: number, incompleteRecoil: boolean) => void = () => {};

  setCallback(callback: (bpm: number, depth: number, incompleteRecoil: boolean) => void) {
    this.onCompressionDetected = callback;
  }

  public calibrate() {
    this.beatTimestamps = [];
    this.lastBeatTime = 0;
    hilog.info(0x0000, 'HACKATHON', "Sensor Calibrated");
  }

  start() {
    if (this.isListening) return;
    this.isListening = true;
    this.beatTimestamps = [];
    this.lastBeatTime = 0;

    this.startWatchdog();

    if (!this.SIMULATION_MODE) {
      this.startRealSensors();
    }
  }

  stop() {
    this.isListening = false;
    this.stopRealSensors();
    clearInterval(this.inactivityTimer);
  }

  private startWatchdog() {

    this.inactivityTimer = setInterval(() => {
      if (!this.isListening) return;

      let now = Date.now();

      this.beatTimestamps = this.beatTimestamps.filter(t => now - t < this.WINDOW_SIZE_MS);

      if (this.lastBeatTime > 0 && (now - this.lastBeatTime > 1500)) {

        this.onCompressionDetected(0, 0, false);
      }

      else if (this.beatTimestamps.length >= 2) {
        let bpm = this.calculateWindowedBPM();

      }
    }, 1000);
  }

  private startRealSensors() {
    try {
      sensor.on(sensor.SensorId.ACCELEROMETER, (data) => {

        this.updateSmoothBuffer(this.bufX, data.x);
        this.updateSmoothBuffer(this.bufY, data.y);
        this.updateSmoothBuffer(this.bufZ, data.z);

        let sx = this.getAvg(this.bufX);
        let sy = this.getAvg(this.bufY);
        let sz = this.getAvg(this.bufZ);

        this.processCPRPulse(sx, sy, sz);
      }, { interval: 20000000 });

    } catch (e) {
      hilog.error(0x0000, 'HACKATHON', "Sensor Start Error: " + JSON.stringify(e));
    }
  }

  private updateSmoothBuffer(buf: number[], val: number) {
    buf.push(val);
    buf.shift();
  }

  private getAvg(buf: number[]): number {
    return (buf[0] + buf[1] + buf[2]) / 3;
  }

  private processCPRPulse(x: number, y: number, z: number) {
    let magnitude = Math.sqrt(x*x + y*y + z*z);

    this.gravityEstimate = 0.995 * this.gravityEstimate + 0.005 * magnitude;

    let energy = Math.abs(magnitude - this.gravityEstimate);

    let now = Date.now();

    if (energy > this.PUSH_THRESHOLD) {

      if (now - this.lastBeatTime > this.MIN_BEAT_INTERVAL) {

        this.beatTimestamps.push(now);
        this.lastBeatTime = now;

        this.beatTimestamps = this.beatTimestamps.filter(t => now - t < this.WINDOW_SIZE_MS);

        let stableBPM = this.calculateWindowedBPM();

        let estimatedDepth = (energy / 11.0) * 5.5;
        if (estimatedDepth < 2) estimatedDepth = 2;
        if (estimatedDepth > 6.5) estimatedDepth = 6.5;

        this.onCompressionDetected(stableBPM, estimatedDepth, false);
      }
    }
  }

  private calculateWindowedBPM(): number {
    if (this.beatTimestamps.length < 2) {

      return this.beatTimestamps.length === 1 ? 110 : 0;
    }

    let first = this.beatTimestamps[0];
    let last = this.beatTimestamps[this.beatTimestamps.length - 1];
    let durationSeconds = (last - first) / 1000.0;
    let count = this.beatTimestamps.length - 1;

    if (durationSeconds <= 0) return 0;

    let bpm = (count / durationSeconds) * 60;

    if (bpm > 200) bpm = 200;

    return bpm;
  }

  private stopRealSensors() {
    try {
      sensor.off(sensor.SensorId.ACCELEROMETER);
    } catch (e) {}
  }
}