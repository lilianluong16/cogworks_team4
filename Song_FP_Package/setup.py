from setuptools import setup, find_packages


def do_setup():
    setup(name='Song_FP',
          version="0.0",
          author='Lilian Luong, Ameer Syedibrahim, Jaden Tennis',
          description='Face database and identifying using Dlib',
          platforms=['Windows', 'Linux', 'Mac OS-X', 'Unix'],
          packages=find_packages())

if __name__ == "__main__":
    do_setup()
    