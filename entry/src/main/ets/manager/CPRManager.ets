import { FeedbackService } from '../services/FeedbackService';
import { AudioService } from '../services/AudioService';
import { CPRStats } from '../model/Models';

export class CPRManager {
  private haptics: FeedbackService;
  private audio: AudioService;
  public stats: CPRStats = new CPRStats();

  public feedbackText: string = "PERFECT";

  private isRunning: boolean = false;
  private metronomeTimer: number = 0;

  private lastErrorTime: number = 0;
  private lastPositiveTime: number = 0;

  private uiUpdateCallback: (stats: CPRStats, text: string) => void = () => {};

  constructor() {
    this.haptics = new FeedbackService();
    this.audio = new AudioService();
  }

  setUIUpdateCallback(cb: (stats: CPRStats, text: string) => void) {
    this.uiUpdateCallback = cb;
  }

  public ingestAIResults(bpm: number, depth: number) {
    this.stats.currentBPM = bpm;
    this.stats.lastPushDepth = depth;
    this.stats.totalCompressions++;

    this.decideFeedback(bpm, depth);

    this.uiUpdateCallback(this.stats, this.feedbackText);
  }

  private decideFeedback(bpm: number, depth: number) {
    let now = Date.now();

    if (now - this.lastErrorTime < 2000) return;

    if (bpm > 125) {
      this.handleError("TOO FAST", "slower.mp3");
      return;
    }
    if (bpm < 100) {
      this.handleError("TOO SLOW", "faster.mp3");
      return;
    }

    if (depth < 5.0) {
      this.handleError("PUSH DEEPER", "deeper.mp3");
      return;
    }
    if (depth > 6.0) {
      this.handleError("PUSH LIGHTER", "lighter.mp3");
      return;
    }

    this.feedbackText = "PERFECT RHYTHM";

    if (now - this.lastPositiveTime > 15000) {
      this.audio.playVoiceCommand("youreDoingGreat.mp3");
      this.lastPositiveTime = now;
    }
  }

  private handleError(text: string, audioFile: string) {
    this.feedbackText = text;
    this.haptics.triggerError();
    this.audio.playVoiceCommand(audioFile);
    this.lastErrorTime = Date.now();
  }

  startCPR() {
    this.isRunning = true;
    this.stats.startTime = Date.now();
    this.stats.totalCompressions = 0;

    this.audio.playVoiceCommand("startingCPRassist.mp3");

    this.metronomeTimer = setInterval(() => {
      if(this.isRunning) this.haptics.triggerMetronome();
    }, 545);
  }

  stopCPR() {
    this.isRunning = false;
    clearInterval(this.metronomeTimer);
  }
}