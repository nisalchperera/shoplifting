import ffmpeg

from glob import glob

video_files = (glob(f"data/*/*.mkv"))
input_file = 'input.webm'
output_file = 'output.mp4'

for input_file in video_files:
    try:
        (
            ffmpeg
            .input(input_file)
            .output(input_file.replace(".mkv", ".mp4"), codec='copy')
            .overwrite_output()
            .run()
        )
        print(f"Successfully converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")