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
    def concatenateVideo(videos):
        '''
         Метод обьединяющий несколько видео в один видеопоток
        '''
        concatenate_video = concatenate_videoclips(videos)

        return concatenate_video
    
    @staticmethod
    def addPreview(preview, recap):
        '''
        Метод добавляющий превью "В предыдущих сериях..." в начало видео
        '''
        full_recap = concatenate_videoclips([preview, recap])

        return full_recap