export class CprFeatureExtractor {
  // Raw Data Buffers
  private xBuffer: number[] = [];
  private yBuffer: number[] = [];
  private zBuffer: number[] = [];
  private gyroBuffer: number[] = [];
  
  private currentHr: number = 80;

  // THE HOOK: Called when 1 second of data is ready
  private onWindowReady: (x: number[], y: number[], z: number[], gyro: number[], hr: number) => void;

  constructor(onWindowReady: (x: number[], y: number[], z: number[], gyro: number[], hr: number) => void) {
    this.onWindowReady = onWindowReady;
  }

  // Call this 50 times a second from Accelerometer
  public addAccelData(x: number, y: number, z: number) {
    this.xBuffer.push(x);
    this.yBuffer.push(y);
    this.zBuffer.push(z);

    // TRIGGER: When we have ~50 samples (1 second), process it.
    if (this.zBuffer.length >= 50) {
      this.flush();
    }
  }

  // Call this 50 times a second from Gyro
  public addGyroData(x: number) {
    this.gyroBuffer.push(x); // We mostly care about X-axis rocking
  }

  // Call this 1 time a second from Heart Rate
  public updateHeartRate(hr: number) {
    this.currentHr = hr;
  }

  private flush() {
    // Safety Check: Ignore empty windows (e.g. app start)
    if (this.zBuffer.length > 5) {
        // Send COPIES of the data to the Brain
        this.onWindowReady(
            [...this.xBuffer], 
            [...this.yBuffer], 
            [...this.zBuffer], 
            [...this.gyroBuffer], 
            this.currentHr
        );
    }

    // Clear buffers for the next second
    this.xBuffer = [];
    this.yBuffer = [];
    this.zBuffer = [];
    this.gyroBuffer = [];
  }
}