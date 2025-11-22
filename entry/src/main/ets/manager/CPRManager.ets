import { FeedbackService } from '../services/FeedbackService';
import { AudioService } from '../services/AudioService';
import { CPRStats } from '../model/Models';
import { SensorService } from '../services/SensorService';
import { common } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

export class CPRManager {
  private haptics: FeedbackService;
  private audio: AudioService;
  public stats: CPRStats = new CPRStats();

  public feedbackText: string = "PREPARE";
  private isRunning: boolean = false;
  private metronomeTimer: number = 0;

  private lastErrorTime: number = 0;
  private lastPositiveTime: number = 0;

  private ERROR_COOLDOWN: number = 5000;

  private shallowPushCounter: number = 0;
  private erraticRhythmCounter: number = 0;
  private FATIGUE_THRESHOLD: number = 10;

  private uiUpdateCallback: (stats: CPRStats, text: string) => void = () => {};

  constructor(context: common.UIAbilityContext) {
    this.haptics = new FeedbackService();
    this.audio = new AudioService(context);
  }

  setUIUpdateCallback(cb: (stats: CPRStats, text: string) => void) {
    this.uiUpdateCallback = cb;
  }

  public calibrateSensor(sensorService: SensorService) {
    sensorService.calibrate();
    this.feedbackText = "CALIBRATED";
    this.haptics.triggerMetronome();
    this.uiUpdateCallback(this.stats, "READY");
  }

  public ingestAIResults(bpm: number, depth: number, incompleteRecoil: boolean) {
    this.stats.currentBPM = bpm;
    this.stats.lastPushDepth = depth;
    if (bpm > 0) this.stats.totalCompressions++;

    this.decideFeedback(bpm, depth);
    this.uiUpdateCallback(this.stats, this.feedbackText);
  }

  private decideFeedback(bpm: number, depth: number) {
    let now = Date.now();

    if (bpm === 0) {
      if (now - this.lastErrorTime > 3000) {
        this.handleError("PRESS FASTER", "faster.mp3");
      }
      return;
    }

    if (now - this.lastErrorTime < this.ERROR_COOLDOWN) return;

    if (bpm >= 100 && bpm <= 120) {

      if (now - this.lastPositiveTime > 12000) {
        this.feedbackText = "PERFECT RHYTHM";
        this.audio.playVoiceCommand("youreDoingGreat.mp3");
        this.lastPositiveTime = now;
      } else {
        this.feedbackText = "PERFECT RHYTHM";
      }

      return;
    }

    if (bpm > 120) {
      this.handleError("PRESS SLOWER", "slower.mp3");
      return;
    }
    if (bpm < 100) {
      this.handleError("PRESS FASTER", "faster.mp3");
      return;
    }

  }

  private handleError(text: string, audioFile: string) {
    this.feedbackText = text;
    this.haptics.triggerError();
    this.audio.playVoiceCommand(audioFile);
    this.lastErrorTime = Date.now();
  }

  startCPR(sensorService: SensorService) {
    this.isRunning = true;
    this.stats.startTime = Date.now();
    this.stats.totalCompressions = 0;
    this.stats.currentBPM = 0;

    hilog.info(0x0000, 'HACKATHON', "Starting CPR Sequence");

    this.audio.playVoiceCommand("startingCPRassist.mp3");

    setTimeout(() => {
      if(this.isRunning) {
        this.calibrateSensor(sensorService);
      }
    }, 4000);

    this.metronomeTimer = setInterval(() => {
      if(this.isRunning) this.haptics.triggerMetronome();
    }, 545);
  }

  stopCPR() {
    this.isRunning = false;
    clearInterval(this.metronomeTimer);
  }
}