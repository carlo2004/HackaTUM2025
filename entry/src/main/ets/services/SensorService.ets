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

  private SIMULATION_MODE: boolean = true;

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

  private onCompressionDetected: (force: number) => void = () => {};

  setCallback(callback: (force: number) => void) {
    this.onCompressionDetected = callback;
  }

  public getAccelX(): number { return this.latestAccelX; }
  public getAccelY(): number { return this.latestAccelY; }
  public getAccelZ(): number { return this.latestAccelZ; }

  public getGyroX(): number { return this.latestGyroX; }
  public getGyroY(): number { return this.latestGyroY; }
  public getGyroZ(): number { return this.latestGyroZ; }

  public getHeartRate(): number { return this.latestHR; }
  public getOxygen(): number { return this.latestSpO2; }
  public getTemperature(): number { return this.latestTemp; }

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

        if (data.z < -15) {
          this.onCompressionDetected(data.z);
          hilog.warn(0x0000, 'HACKATHON', "CPR COMPRESSION DETECTED: %{public}s", data.z.toFixed(2));
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

        this.logSensor("Real_HR", data.heartRate);
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

  private startSimulation() {
    let tick = 0;

    this.simTimer = setInterval(() => {
      tick += 0.1;

      let pushForce = -9.8 + (Math.sin(tick) * 12);
      if (pushForce < -15) {
        this.latestAccelZ = pushForce;
        this.onCompressionDetected(this.latestAccelZ);
      } else {
        this.latestAccelZ = -9.8;
      }

      this.latestAccelX = (Math.random() - 0.5) * 2;
      this.latestAccelY = (Math.random() - 0.5) * 2;

      this.latestGyroX = Math.sin(tick * 0.5) * 0.5;
      this.latestGyroY = Math.cos(tick * 0.5) * 0.5;
      this.latestGyroZ = (Math.random() - 0.5) * 0.1;

      this.latestHR = 80 + Math.floor(tick * 2);
      if (this.latestHR > 160) this.latestHR = 160;

      this.logSensor("Sim_Acc_X", this.latestAccelX);
      this.logSensor("Sim_Acc_Y", this.latestAccelY);
      this.logSensor("Sim_Acc_Z", this.latestAccelZ);

      this.logSensor("Sim_Gyr_X", this.latestGyroX);
      this.logSensor("Sim_Gyr_Y", this.latestGyroY);
      this.logSensor("Sim_Gyr_Z", this.latestGyroZ);

      this.logSensor("Sim_HR", this.latestHR);

    }, 100);
  }
}