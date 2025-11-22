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

  private MIN_BEAT_INTERVAL: number = 440;

  private PUSH_THRESHOLD: number = 2.0;

  private WEAK_THRESHOLD: number = 1.7;

  private mlAccX: number[] = [];
  private mlAccY: number[] = [];
  private mlAccZ: number[] = [];
  private mlGyrX: number[] = [];
  private mlGyrY: number[] = [];
  private mlGyrZ: number[] = [];
  private mlHR: number[] = [];
  private ML_WINDOW_SIZE: number = 75;

  private onCompressionDetected: (bpm: number, depth: number, incompleteRecoil: boolean, mlFeedback: string) => void = () => {};

  setCallback(callback: (bpm: number, depth: number, incompleteRecoil: boolean, mlFeedback: string) => void) {
    this.onCompressionDetected = callback;
  }

  public calibrate() {
    this.beatTimestamps = [];
    this.lastBeatTime = 0;
    this.clearMLBuffers();
    hilog.info(0x0000, 'HACKATHON', "Sensor Calibrated");
  }

  start() {
    if (this.isListening) return;
    this.isListening = true;
    this.beatTimestamps = [];
    this.lastBeatTime = 0;
    this.clearMLBuffers();
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

      if (this.mlAccX.length >= this.ML_WINDOW_SIZE) {
        this.processMLWindow();
      }

      if (this.lastBeatTime > 0 && (now - this.lastBeatTime > 1500)) {
        this.onCompressionDetected(0, 0, false, "NONE");
      }
    }, 100);
  }

  private startRealSensors() {
    try {
      sensor.on(sensor.SensorId.ACCELEROMETER, (data) => {
        this.mlAccX.push(data.x);
        this.mlAccY.push(data.y);
        this.mlAccZ.push(data.z);

        this.updateSmoothBuffer(this.bufX, data.x);
        this.updateSmoothBuffer(this.bufY, data.y);
        this.updateSmoothBuffer(this.bufZ, data.z);

        let sx = this.getAvg(this.bufX);
        let sy = this.getAvg(this.bufY);
        let sz = this.getAvg(this.bufZ);

        this.processCPRPulse(sx, sy, sz);
      }, { interval: 20000000 });

      sensor.on(sensor.SensorId.GYROSCOPE, (data) => {
        this.mlGyrX.push(data.x);
        this.mlGyrY.push(data.y);
        this.mlGyrZ.push(data.z);
      }, { interval: 20000000 });

      sensor.on(sensor.SensorId.HEART_RATE, (data) => {
        this.mlHR.push(data.heartRate);
      }, { interval: 1000000000 });

    } catch (e) {
      hilog.error(0x0000, 'HACKATHON', "Sensor Start Error: " + JSON.stringify(e));
    }
  }

  private processMLWindow() {
    try {
      let features = this.extractFeatures();
      let probs = this.runDecisionTree(features);

      let mlFeedback = "NONE";

      if (probs[1] > 0.65) mlFeedback = "PUSH_HARDER";
      else if (probs[2] > 0.65) mlFeedback = "RELEASE_CHEST";
      else if (probs[0] > 0.65) mlFeedback = "GOOD";

      let minZ = Math.min(...this.mlAccZ);
      if (minZ > 10.5) {
        mlFeedback = "RELEASE_CHEST";
      }

      let currentBPM = this.calculateWindowedBPM();
      this.onCompressionDetected(currentBPM, 0, false, mlFeedback);

    } catch (e) {
      hilog.error(0x0000, 'HACKATHON', "ML Error");
    } finally {
      this.clearMLBuffers();
    }
  }

  private extractFeatures(): number[] {
    const getMean = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const getStd = (arr: number[]) => {
      if (arr.length === 0) return 0;
      let m = getMean(arr);
      let v = arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length;
      return Math.sqrt(v);
    };
    const getMax = (arr: number[]) => arr.length > 0 ? Math.max(...arr) : 0;

    let std_dev_z = getStd(this.mlAccZ);
    let recoil_proxy_z = getMax(this.mlAccZ);
    let recoil_proxy_x = 0;
    let recoil_proxy_y = getMax(this.mlAccX);
    let recoil_proxy = 0;
    let rescuer_hr = getMean(this.mlHR);

    let mean_x = getMean(this.mlAccX);
    let mean_y = getMean(this.mlAccY);
    let tilt_error = Math.sqrt(mean_x * mean_x + mean_y * mean_y);

    let wobble_score_x = getStd(this.mlGyrX);
    let wobble_score_y = getStd(this.mlGyrY);
    let wobble_score_z = getStd(this.mlGyrZ);

    return [
      std_dev_z, recoil_proxy_z, recoil_proxy_x, recoil_proxy_y, recoil_proxy,
      rescuer_hr, tilt_error, wobble_score_x, wobble_score_y, wobble_score_z, 0
    ];
  }

  private runDecisionTree(input: number[]): number[] {
    let var0: number[] = [0, 0, 0];
    if (input[7] <= 0.21676643192768097) {
      if (input[9] <= 0.11362288147211075) {
        if (input[7] <= 0.11024455726146698) {
          var0 = [0.05555555555555555, 0.7777777777777778, 0.16666666666666666];
        } else {
          if (input[5] <= 37.0) {
            var0 = [0.0, 0.7142857142857143, 0.2857142857142857];
          } else {
            var0 = [0.0, 0.15789473684210525, 0.8421052631578947];
          }
        }
      } else {
        if (input[5] <= 76.75) {
          var0 = [0.5, 0.0, 0.5];
        } else {
          var0 = [0.0, 0.96, 0.04];
        }
      }
    } else {
      if (input[6] <= 9.231021404266357) {
        var0 = [0.25, 0.35, 0.4];
      } else {
        if (input[8] <= 0.10734017938375473) {
          var0 = [0.0, 0.14285714285714285, 0.8571428571428571];
        } else {
          var0 = [0.9, 0.0625, 0.0375];
        }
      }
    }
    return var0;
  }

  private clearMLBuffers() {
    this.mlAccX = [];
    this.mlAccY = [];
    this.mlAccZ = [];
    this.mlGyrX = [];
    this.mlGyrY = [];
    this.mlGyrZ = [];
    this.mlHR = [];
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

    if (energy > this.WEAK_THRESHOLD) {

      if (now - this.lastBeatTime > this.MIN_BEAT_INTERVAL) {

        this.beatTimestamps.push(now);
        this.lastBeatTime = now;
        this.beatTimestamps = this.beatTimestamps.filter(t => now - t < this.WINDOW_SIZE_MS);

        let stableBPM = this.calculateWindowedBPM();

        let estimatedDepth = 5.0;

        if (energy < this.PUSH_THRESHOLD) {
          estimatedDepth = 2.5;
        } else {

          estimatedDepth = (energy / 11.0) * 5.5;
          if (estimatedDepth > 6.5) estimatedDepth = 6.5;
        }

        let manualFeedback = (estimatedDepth < 4.0) ? "PUSH_HARDER" : "NONE";

        this.onCompressionDetected(stableBPM, estimatedDepth, false, manualFeedback);
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
      sensor.off(sensor.SensorId.GYROSCOPE);
      sensor.off(sensor.SensorId.HEART_RATE);
    } catch (e) {}
  }
}