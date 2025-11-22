import { CPRManager } from '../manager/CPRManager';
import { CPRStats } from '../model/Models';
import { SensorService } from '../services/SensorService';

@Entry
@Component
struct Index {
  private cprManager: CPRManager = new CPRManager();

  private sensors: SensorService = new SensorService();

  @State feedbackMsg: string = "START CPR";
  @State isRunning: boolean = false;
  @State bgColor: string = '#000000';

  aboutToAppear() {
    this.cprManager.setUIUpdateCallback((_stats: CPRStats, text: string) => {      this.feedbackMsg = text;

      if (text.includes("PERFECT")) {
        this.bgColor = '#003300';
      } else {
        this.bgColor = '#330000';
      }
    });
  }

  build() {
    Column() {

      Text(this.feedbackMsg)
        .fontSize(40)
        .fontWeight(FontWeight.Bolder)
        .textAlign(TextAlign.Center)
        .fontColor(Color.White)
        .margin({ top: 40 })

      Circle({ width: 100, height: 100 })
        .fill(Color.White)
        .opacity(0.2)
        .animation({ duration: 545, iterations: -1, curve: Curve.EaseInOut })
        .margin(20)

      Button(this.isRunning ? "STOP" : "START")
        .width('80%')
        .height(60)
        .backgroundColor('#E60000')
        .onClick(() => {
          this.isRunning = !this.isRunning;

          if (this.isRunning) {

            this.cprManager.startCPR();
            this.feedbackMsg = "LISTEN TO RHYTHM";

            this.sensors.start();
            console.error("🚨 UI: STARTING SENSORS MANUALLY 🚨");

          } else {

            this.cprManager.stopCPR();
            this.sensors.stop();
            this.feedbackMsg = "SESSION ENDED";
            this.bgColor = '#000000';
          }
        })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(this.bgColor)
    .animation({ duration: 300 })
  }
}