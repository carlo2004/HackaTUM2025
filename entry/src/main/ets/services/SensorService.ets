import { sensor } from '@kit.SensorServiceKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

export class SensorBundle {
  accelZ: number = 0;
  gyroX: number = 0;
  heartRate: number = 0;
  oxygen: number = 0;
  temperature: number = 0;
}

export class SensorService {

  private SIMULATION_MODE: boolean = true;

  private latestZ: number = 0;
  private latestGyroX: number = 0;
  private latestHR: number = 0;
  private latestSpO2: number = 98;
  private latestTemp: number = 36.5;

  private isListening: boolean = false;
  private simTimer: number = 0;

  private onCompressionDetected: (force: number) => void = () => {};

  setCallback(callback: (force: number) => void) {
    this.onCompressionDetected = callback;
  }

  public getAccelZ(): number { return this.latestZ; }
  public getGyroX(): number { return this.latestGyroX; }
  public getHeartRate(): number { return this.latestHR; }
  public getOxygen(): number { return this.latestSpO2; }
  public getTemperature(): number { return this.latestTemp; }

  public getAllSensors(): SensorBundle {
    let bundle = new SensorBundle();
    bundle.accelZ = this.latestZ;
    bundle.gyroX = this.latestGyroX;
    bundle.heartRate = this.latestHR;
    bundle.oxygen = this.latestSpO2;
    bundle.temperature = this.latestTemp;
    return bundle;
  }

  private logSensor(name: string, val: number) {

    hilog.info(0x0000, 'HACKATHON', '%{public}s: %{public}s', name, val.toFixed(4));  }

  start() {
    if (this.isListening) return;
    this.isListening = true;

    if (this.SIMULATION_MODE) {
      console.error("🚨 STARTING SIMULATION 🚨");
      this.startSimulation();
    } else {
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
  }

  private startRealSensors() {
    try {
      sensor.on(sensor.SensorId.ACCELEROMETER, (data) => {
        this.latestZ = data.z;
        if (data.z < -15) this.onCompressionDetected(data.z);
      }, { interval: 20000000 });

      sensor.on(sensor.SensorId.GYROSCOPE, (data) => {
        this.latestGyroX = data.x;
      }, { interval: 60000000 });

      sensor.on(sensor.SensorId.HEART_RATE, (data) => {
        this.latestHR = data.heartRate;
      }, { interval: 1000000000 });
    } catch (e) {
      console.error("Real Sensor Error: " + JSON.stringify(e));
    }
  }

  private stopRealSensors() {
    try {
      sensor.off(sensor.SensorId.ACCELEROMETER);
      sensor.off(sensor.SensorId.GYROSCOPE);
      sensor.off(sensor.SensorId.HEART_RATE);
    } catch (e) {}
  }

  private startSimulation() {
    let tick = 0;

    this.simTimer = setInterval(() => {
      tick += 0.1;

      let pushForce = -9.8 + (Math.sin(tick) * 12);
      if (pushForce < -15) {
        this.latestZ = pushForce;
        this.onCompressionDetected(this.latestZ);
      } else {
        this.latestZ = -9.8;
      }

      this.latestHR = 80 + Math.floor(tick * 2);
      if (this.latestHR > 160) this.latestHR = 160;
      this.latestGyroX = (Math.random() - 0.5) * 0.5;

      this.logSensor("Accel_Z", this.latestZ);
      this.logSensor("Gyro_X", this.latestGyroX);
      this.logSensor("HR", this.latestHR);

    }, 100);
  }
}