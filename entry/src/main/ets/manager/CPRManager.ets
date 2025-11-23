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

  private activeMLError: string = "NONE";
  private mlErrorCounter: number = 0;

  private REQUIRED_CONSISTENCY: number = 3;

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

  public ingestAIResults(bpm: number, depth: number, incompleteRecoil: boolean, mlFeedback: string) {
    this.stats.currentBPM = bpm;

    if (bpm > 0) {
      this.stats.lastPushDepth = depth;
      this.stats.totalCompressions++;
    }

    let currentError = mlFeedback;

    if (currentError === "PUSH_HARDER" || currentError === "RELEASE_CHEST") {
      if (currentError === this.activeMLError) {
        this.mlErrorCounter++;
      } else {
        this.activeMLError = currentError;
        this.mlErrorCounter = 1;
      }
    } else {
      this.mlErrorCounter = 0;
      this.activeMLError = "NONE";
    }

    this.decideFeedback(bpm);
    this.uiUpdateCallback(this.stats, this.feedbackText);
  }

  private decideFeedback(bpm: number) {
    let now = Date.now();
    let visualState = "ANALYZING...";

    if (bpm === 0) {
      visualState = "PRESS FASTER";
    }

    else if (this.mlErrorCounter >= this.REQUIRED_CONSISTENCY) {
      if (this.activeMLError === "PUSH_HARDER") {
        visualState = "PUSH DEEPER";
      } else if (this.activeMLError === "RELEASE_CHEST") {
        visualState = "RELEASE CHEST";
      }
    }

    else {
      visualState = this.getRhythmState(bpm);
    }

    this.feedbackText = visualState;
    this.processAudio(visualState, bpm);
  }

  private getRhythmState(bpm: number): string {
    if (bpm >= 100 && bpm <= 120) return "PERFECT RHYTHM";
    if (bpm > 120) return "PRESS SLOWER";
    return "PRESS FASTER";
  }

  private processAudio(visualState: string, bpm: number) {
    let now = Date.now();

    if (bpm === 0) {
      if (now - this.lastErrorTime > 3000) {
        this.triggerAudio("PRESS FASTER", "faster.mp3");
      }
      return;
    }

    if (now - this.lastErrorTime < this.ERROR_COOLDOWN) return;

    if (visualState === "PERFECT RHYTHM") {
      if (now - this.lastPositiveTime > 12000) {
        this.audio.playVoiceCommand("youreDoingGreat.mp3");
        this.lastPositiveTime = now;
      }
      return;
    }

    if (visualState === "PRESS SLOWER") this.triggerAudio("PRESS SLOWER", "slower.mp3");
    else if (visualState === "PRESS FASTER") this.triggerAudio("PRESS FASTER", "faster.mp3");
    else if (visualState === "PUSH DEEPER") this.triggerAudio("PUSH DEEPER", "pushDeeper.mp3");
    else if (visualState === "RELEASE CHEST") this.triggerAudio("RELEASE CHEST", "releaseChestFully.mp3");
  }

  private triggerAudio(text: string, audioFile: string) {
    this.haptics.triggerError();
    this.audio.playVoiceCommand(audioFile);
    this.lastErrorTime = Date.now();
  }

  startCPR(sensorService: SensorService) {
    this.isRunning = true;
    this.stats.startTime = Date.now();
    this.stats.totalCompressions = 0;
    this.stats.currentBPM = 0;
    this.activeMLError = "NONE";
    this.mlErrorCounter = 0;

    hilog.info(0x0000, 'HACKATHON', "Starting CPR Sequence");

    this.audio.playVoiceCommand("startingCPRassist.mp3");

    setTimeout(() => {
      if(this.isRunning) {
        this.calibrateSensor(sensorService);
      }
    }, 5000);

    this.metronomeTimer = setInterval(() => {
      if(this.isRunning) this.haptics.triggerMetronome();
    }, 545);
  }

  stopCPR() {
    this.isRunning = false;
    clearInterval(this.metronomeTimer);
  }
}