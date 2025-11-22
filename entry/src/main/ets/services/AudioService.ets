import { media } from '@kit.MediaKit';
import { common } from '@kit.AbilityKit';

export class AudioService {
  private avPlayer: media.AVPlayer | undefined;
  private context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;

  async playVoiceCommand(filename: string) {

    if (this.avPlayer && this.avPlayer.state === 'playing') return;

    try {

      this.avPlayer = await media.createAVPlayer();

      let fileDescriptor = await this.context.resourceManager.getRawFd(filename);
      this.avPlayer.fdSrc = fileDescriptor;

      this.avPlayer.on('stateChange', (state) => {
        if (state === 'initialized') {
          this.avPlayer?.play();
        } else if (state === 'completed') {
          this.avPlayer?.release();
        }
      });
    } catch (e) {
      console.error("Audio Failed: " + e);
    }
  }
}