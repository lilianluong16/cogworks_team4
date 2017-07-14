""""This Song class is useful for storing songs in our database as objects and allows for quick retrieval of the song's name,
artist and path


NOTE: When creating an instance of the Song class, an 'r' must be placed in front of the song path, which is the third parameter
in the initialization function
 """

class Song:
    def __init__(self,nm,artst,sng_pth=None):
        self.name = nm
        self.artist = artst
        self.song_path = sng_pth
    
    def __repr__(self):
        return self.name
    
    def show_Artist(self):
        return self.artist
    
    def show_Song_Path(self):
        return self.song_path



