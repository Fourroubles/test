from moviepy import VideoFileClip, concatenate_videoclips

class VideoProcessing:
    @staticmethod
    def cutVideo(path_video, left_border, right_border):
        '''
        Метод обрезающий видео в заданных границах
        '''
        recap = VideoFileClip(path_video).subclipped(left_border, right_border)
  
        return recap
    
    @staticmethod
    def saveVideo(path_folder, video):
        '''
        Метод сохраняющий видео в деректорию
        '''
        video.write_videofile(path_folder)

    @staticmethod
    def concatenateVideo(path_videos):
        '''
         Метод обьединяющий несколько видео в один видеопоток
        '''
        tmp_video = []
        for video in path_videos:
            tmp_video.append(VideoFileClip(video))

        concatenate_video = concatenate_videoclips(tmp_video)

        return concatenate_video
    
    @staticmethod
    def addPreview(preview, recap):
        '''
        Метод добавляющий превью "В предыдущих сериях..." в начало видео
        '''
        full_recap = concatenate_videoclips([preview, recap])

        return full_recap