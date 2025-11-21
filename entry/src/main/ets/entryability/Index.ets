// src/main/ets/pages/Index.ets
import { CPRManager } from '../manager/CPRManager';
import { CPRStats } from '../model/Models';

@Entry
@Component
struct Index {
  // 1. Instantiate the Manager
  private cprManager: CPRManager = new CPRManager();

  // 2. UI State Variables
  @State compressions: number = 0;
  @State isRunning: boolean = false;

  aboutToAppear() {
    // 3. Listen for updates from the Manager
    this.cprManager.setUIUpdateCallback((newStats: CPRStats) => {
      this.compressions = newStats.totalCompressions;
    });
  }

  build() {
    Column() {
      // The Pulse Animation Circle
      Circle({ width: 150, height: 150 })
        .fill(this.isRunning ? Color.Red : Color.Gray)
        .animation({ duration: 500, curve: Curve.EaseInOut, iterations: -1 })

      Text(this.compressions.toString())
        .fontSize(60)
        .fontWeight(FontWeight.Bold)
        .fontColor(Color.White)
        .margin(20)

      Button(this.isRunning ? "STOP" : "START CPR")
        .width('80%')
        .height(60)
        .backgroundColor(this.isRunning ? Color.Gray : '#E60000')
        .onClick(() => {
          this.isRunning = !this.isRunning;
          if (this.isRunning) {
            this.cprManager.startCPR();
          } else {
            this.cprManager.stopCPR();
          }
        })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    .justifyContent(FlexAlign.Center)
  }
}