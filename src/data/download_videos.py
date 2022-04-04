from pytube import YouTube

with open('./links.csv','r') as in_file:
    next(in_file)
    for i, line in enumerate(in_file):
        link = line.strip()
        try:
            yt = YouTube(link)
            d_video = yt.streams.filter(file_extension='mp4',adaptive=True, res='360p').first()
            d_video.download(output_path='videos', filename=f'{i}_{d_video.resolution}_{d_video.fps}.mp4')
        except Exception:
            print(f"Some Error at video {i}")