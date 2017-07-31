from setuptools import setup, find_packages


def do_setup():
    setup(name='News_Buddy',
          version="0.0",
          author='Lilian Luong, Ameer Syedibrahim, Jaden Tennis',
          description='News database by topic and named entities',
          platforms=['Windows', 'Linux', 'Mac OS-X', 'Unix'],
          packages=find_packages())

if __name__ == "__main__":
    do_setup()
    