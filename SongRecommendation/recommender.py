"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


import json
import os
from pathlib import Path
from typing import Tuple, List

import implicit
import psutil
import scipy

import scipy
import pandas as pd

def load_user_songs(user_songs_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user songs file and return a user-songs matrix in csr
    fromat.
    """
    user_songs = pd.read_csv(user_songs_file, sep="\t")
    user_songs.set_index(["userId", "songNo"], inplace=True)
    
    coo = scipy.sparse.coo_matrix(
        (
            user_songs.weight.astype(float),
            (
                user_songs.index.get_level_values(0),
                user_songs.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()

def load_user_artist(user_artist_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user songs file and return a user-songs matrix in csr
    fromat.
    """
    user_artist = pd.read_csv(user_artist_file, sep="\t")
    user_artist.set_index(["userId", "artistNo"], inplace=True)
    
    coo = scipy.sparse.coo_matrix(
        (
            user_artist.weight.astype(float),
            (
                user_artist.index.get_level_values(0),
                user_artist.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()


class TrackRetriever:
    """The TrackRetriever class gets the song name from the song ID."""

    def __init__(self):
        self.num_to_track_id = None
        self.track_id_to_num = None

    def get_track_id_from_num(self, number: id) -> str:
        """Return the track id from the number."""
        return self.num_to_track_id.loc[number, 'trackId']
    
    def get_num_from_track_id(self, track_id: str) -> id:
        """Return the number from the track ID."""
        return self.track_id_to_num.loc[track_id, 'no']

    def load_num_to_track_id(self, map_file: Path) -> None:
        """Load the songs file and stores it as a Pandas dataframe in a
        private attribute.
        """
        df = pd.read_csv(map_file, sep="\t")
        df = df.set_index("no")
        self.num_to_track_id = df
        
    def load_track_id_to_num(self, map_file: Path) -> None:
        """Load the songs file and stores it as a Pandas dataframe in a
        private attribute.
        """
        df = pd.read_csv(map_file, sep="\t")
        df = df.set_index("trackId")
        self.track_id_to_num = df


class ArtistRetriever:
    """The ArtistRetriever class gets the song name from the song ID."""

    def __init__(self):
        self.num_to_artist_id = None
        self.artist_id_to_num = None

    def get_artist_id_from_num(self, number: id) -> str:
        """Return the artistId from the number."""
        return self.num_to_artist_id.loc[number, 'artistId']
    
    def get_num_from_artist_id(self, track_id: str) -> id:
        """Return the number from the artistId."""
        return self.artist_id_to_num.loc[track_id, 'no']

    def load_num_to_artist_id(self, map_file: Path) -> None:
        """Load the songs file and stores it as a Pandas dataframe in a
        private attribute.
        """
        df = pd.read_csv(map_file, sep="\t")
        df = df.set_index("no")
        self.num_to_artist_id = df
        
    def load_artist_id_to_num(self, map_file: Path) -> None:
        """Load the songs file and stores it as a Pandas dataframe in a
        private attribute.
        """
        df = pd.read_csv(map_file, sep="\t")
        df = df.set_index("artistId")
        self.artist_id_to_num = df


class ImplicitRecommender:
    """
    The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Props:
        - retriever: TrackRetriever instance 
        - track_implicit_model: an implicit model
    """

    def __init__(
        self,
        track_retriever: TrackRetriever,
        track_implicit_model: implicit.recommender_base.RecommenderBase,
        
        artist_retriever: ArtistRetriever,
        artist_implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.track_retriever = track_retriever
        self.track_implicit_model = track_implicit_model
        self.artist_retriever = artist_retriever
        self.artist_implicit_model = artist_implicit_model

    def fit_track_model(self, user_track_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user songs matrix."""
        self.track_implicit_model.fit(user_track_matrix)
        
    def fit_artist_model(self, user_artist_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user songs matrix."""
        self.artist_implicit_model.fit(user_artist_matrix)

    def recommend_tracks(
        self,
        user_id: int,
        user_track_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        song_ids, scores = self.track_implicit_model.recommend(
            user_id, user_track_matrix[n], N=n
        )
        tracks = [
            self.track_retriever.get_track_id_from_num(song_id)
            for song_id in song_ids
        ]
        return tracks, scores
    
    def recommend_similar_tracks(self, given_track_number, n: int = 10) ->List[str]:
        """Return the top n recommendations for the given user."""
        similar_track_nums, scores = self.track_implicit_model.similar_items(given_track_number, N=n+20)

        # flatten similar_track_nums and scores
        similar_track_nums = [item for sublist in similar_track_nums for item in sublist]
        similar_track_nums = list(dict.fromkeys(similar_track_nums))
        similar_track_nums = sorted(similar_track_nums, key=lambda x: os.urandom(1)) # must be after list(dict.fromkeys(similar_track_nums)) to avoid reordering
            
        # remove given_track_number from similar_track_nums, but dont make len(similar_track_nums) < n
        while len (similar_track_nums) > n:
            found = [item for item in similar_track_nums if item in given_track_number]
            if len(found) > 0:
                print("remove found: ", found[0])
                similar_track_nums.remove(found[0])
            else:
                break

        similar_track_nums_int = [int(item) for item in similar_track_nums]
        print("similar_track_nums_int: ", similar_track_nums_int)
        
        if len(similar_track_nums) > n:
            similar_track_nums = similar_track_nums[:n]
            
        similar_track_nums_int = [int(item) for item in similar_track_nums]
        print("similar_track_nums_int: ", similar_track_nums_int)
        
        
        similar_track_ids = [
            self.track_retriever.get_track_id_from_num(similar_track_num)
            for similar_track_num in similar_track_nums
        ]
        
        return similar_track_ids

    def recommend_artist(
        self,
        user_id: int,
        user_artist_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.artist_implicit_model.recommend(
            user_id, user_artist_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_id_from_num(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores
    
    def recommend_similar_artists(self, given_artist_number: int, n: int = 10) ->List[str]:
        """Return the top n recommendations for the given user."""
        similar_artist_nums, scores = self.artist_implicit_model.similar_items(given_artist_number, N=n+1)
        
        # remove the first element, which is the given_artist_number
        similar_artist_nums = similar_artist_nums[1:]
        
        similar_artist_ids = [
            self.artist_retriever.get_artist_id_from_num(similar_artist_num)
            for similar_artist_num in similar_artist_nums
        ]
        
        return similar_artist_ids
