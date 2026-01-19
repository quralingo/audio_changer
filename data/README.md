To convert the data downloaded from YouTube form mp3 to wave you can run this command

```bash
ffmpeg -y -i "downloaded.mp3" -ac 1 -ar 48000 "converted.wav"
```