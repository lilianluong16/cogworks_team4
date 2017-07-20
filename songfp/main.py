
import audio_record
import database
import fingerprinting
import song_loader


def train_single(filepath):
    s_i = song_loader.mp3_to_samparr(filepath)
    name = song_loader.songpath_to_name(filepath)
    artist = song_loader.songpath_to_artist(filepath)
    spectro_i = audio_record.spectrogram(s_i)
    peaks_i = fingerprinting.find_peaks(spectro_i[0], spectro_i[1])
    features = fingerprinting.find_fingerprint(peaks_i, spectro_i[1], spectro_i[2])
    return name, artist, features


def train(folder="audio"):
    filepaths = song_loader.convert_files_to_songpaths(folder)
    # load songs
    for i in filepaths:
        name, artist, features = train_single(i)
        print("Adding:", name, "by", artist)
        database.add_song(features, name, artist)
        print("Added:", name, "by", artist)
        database.write_database()


def identify():
    song = audio_record.record(10)
    spectro_i = audio_record.spectrogram(song)
    peaks_i = fingerprinting.find_peaks(spectro_i[0], spectro_i[1])
    features = fingerprinting.find_fingerprint(peaks_i, spectro_i[1], spectro_i[2])
    matches = fingerprinting.get_matches(features, database.database)
    best_match = fingerprinting.best_match(matches)
    if best_match is not None:
        print(best_match.name, "by", best_match.artist)
    return best_match


identify()
