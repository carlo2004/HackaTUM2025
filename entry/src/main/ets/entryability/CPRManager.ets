// src/main/ets/manager/CPRManager.ts
import { SensorService } from '../services/SensorService';
import { FeedbackService } from '../services/FeedbackService';
import { CPRStats, PatientProfile } from '../model/Models';

export class CPRManager {
  // Dependencies
  private sensors: SensorService;
  private haptics: FeedbackService;

  // State
  public stats: CPRStats = new CPRStats();
  public profile: PatientProfile = new PatientProfile();
  private lastPushTime: number = 0;
  private isRunning: boolean = false;

  // Callback to update UI
  private uiUpdateCallback: (stats: CPRStats) => void = () => {};

  constructor() {
    this.sensors = new SensorService();
    this.haptics = new FeedbackService();

    // Hook up the sensor to our logic
    this.sensors.setCallback((force: number): void => {
      this.handlePush(force);
    });
  }

  setUIUpdateCallback(cb: (stats: CPRStats) => void) {
    this.uiUpdateCallback = cb;
  }

  startCPR() {
    this.isRunning = true;
    this.stats.startTime = Date.now();
    this.stats.totalCompressions = 0;
    this.sensors.start();

    // Start the Metronome Loop (110 BPM)
    setInterval(() => {
      if(this.isRunning) this.haptics.triggerMetronome();
    }, (60000 / this.profile.targetBPM));
  }

  stopCPR() {
    this.isRunning = false;
    this.sensors.stop();
  }

  // The Core Logic (Debouncing + Counting)
  private handlePush(force: number) {
    let now = Date.now();
    if (now - this.lastPushTime > 300) { // Debounce
      this.lastPushTime = now;
      this.stats.totalCompressions++;
      this.stats.lastPushDepth = force;

      // Calculate real-time BPM based on last 5 pushes (advanced logic here)
      // For now, simplified:
      console.info("Good Compression! Count: " + this.stats.totalCompressions);

      // Notify UI
      this.uiUpdateCallback(this.stats);
    }
  }
}