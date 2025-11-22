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

  private latestAccelX: number = 0;
  private latestAccelY: number = 0;
  private latestAccelZ: number = 0;

  private latestGyroX: number = 0;
  private latestGyroY: number = 0;
  private latestGyroZ: number = 0;

  private latestHR: number = 0;
  private latestSpO2: number = 98;
  private latestTemp: number = 36.5;

  private isListening: boolean = false;
  private simTimer: number = 0;

  private lastCompressionTime: number = 0;
  private currentBPM: number = 0;

  private onCompressionDetected: (bpm: number, depth: number) => void = () => {};

  setCallback(callback: (bpm: number, depth: number) => void) {
    this.onCompressionDetected = callback;
  }

  public getAllSensors(): SensorBundle {
    let bundle = new SensorBundle();
    bundle.accelX = this.latestAccelX;
    bundle.accelY = this.latestAccelY;
    bundle.accelZ = this.latestAccelZ;
    bundle.gyroX = this.latestGyroX;
    bundle.gyroY = this.latestGyroY;
    bundle.gyroZ = this.latestGyroZ;
    bundle.heartRate = this.latestHR;
    bundle.oxygen = this.latestSpO2;
    bundle.temperature = this.latestTemp;
    return bundle;
  }

  private logSensor(name: string, val: number) {
    hilog.info(0x0000, 'HACKATHON', '%{public}s: %{public}s', name, val.toFixed(4));
  }

  start() {
    if (this.isListening) return;
    this.isListening = true;

    if (this.SIMULATION_MODE) {
      hilog.warn(0x0000, 'HACKATHON', "🚨 STARTING SIMULATION MODE 🚨");
      this.startSimulation();
    } else {
      hilog.info(0x0000, 'HACKATHON', "✅ STARTING REAL SENSORS");
      this.startRealSensors();
    }
  }

  stop() {
    this.isListening = false;
    if (this.SIMULATION_MODE) {
      clearInterval(this.simTimer);
    } else {
      this.stopRealSensors();
    }
    hilog.info(0x0000, 'HACKATHON', "🛑 SENSORS STOPPED");
  }

  private startRealSensors() {
    try {

      sensor.on(sensor.SensorId.ACCELEROMETER, (data) => {
        this.latestAccelX = data.x;
        this.latestAccelY = data.y;
        this.latestAccelZ = data.z;

        this.logSensor("Real_Acc_X", data.x);
        this.logSensor("Real_Acc_Y", data.y);
        this.logSensor("Real_Acc_Z", data.z);

        if (data.z < -10) {
          let now = Date.now();

          if (now - this.lastCompressionTime > 300) {

            if (this.lastCompressionTime !== 0) {
              let diff = now - this.lastCompressionTime;
              this.currentBPM = 60000 / diff;
            } else {
              this.currentBPM = 110;
            }

            this.lastCompressionTime = now;

            let estimatedDepth = Math.abs(data.z) / 3.5;
            if (estimatedDepth > 7) estimatedDepth = 6.5;

            this.onCompressionDetected(this.currentBPM, estimatedDepth);

            hilog.warn(0x0000, 'HACKATHON', ">>> COMPRESSION EVENT: BPM %{public}s", this.currentBPM.toFixed(0));
          }
        }
      }, { interval: 20000000 });

      sensor.on(sensor.SensorId.GYROSCOPE, (data) => {
        this.latestGyroX = data.x;
        this.latestGyroY = data.y;
        this.latestGyroZ = data.z;

        this.logSensor("Real_Gyr_X", data.x);
        this.logSensor("Real_Gyr_Y", data.y);
        this.logSensor("Real_Gyr_Z", data.z);
      }, { interval: 60000000 });

      sensor.on(sensor.SensorId.HEART_RATE, (data) => {
        this.latestHR = data.heartRate;

        hilog.info(0x0000, 'HACKATHON', "Real_HR: %{public}s", data.heartRate.toString());

      }, { interval: 1000000000 });

    } catch (e) {
      hilog.error(0x0000, 'HACKATHON', "Real Sensor Error: %{public}s", JSON.stringify(e));
    }
  }

  private stopRealSensors() {
    try {
      sensor.off(sensor.SensorId.ACCELEROMETER);
      sensor.off(sensor.SensorId.GYROSCOPE);
      sensor.off(sensor.SensorId.HEART_RATE);
    } catch (e) {
      hilog.error(0x0000, 'HACKATHON', "Error stopping sensors: %{public}s", JSON.stringify(e));
    }
  }

  private startSimulation() {  }
}