import json
from pathlib import Path
from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import implicit
import pandas as pd

from SongRecommendation.recommender import ArtistRetriever, ImplicitRecommender, TrackRetriever, load_user_artist, load_user_songs
from schema.request import ReqRecommendSimilarArtistSchema, ReqRecommendSimilarTracks

# load user songs matrix D:/DUT/FinalProject/My_Project\SongRecommendation\extracted-data\artist
user_songs = load_user_songs(Path("./extracted-data/track/playlist_track.dat"))
user_artist = load_user_artist(Path("./extracted-data/artist/playlist_artist.dat"))

# instantiate song retriever
track_retriever = TrackRetriever()
track_retriever.load_num_to_track_id(Path("./extracted-data/track/num_to_track_id.dat"))
track_retriever.load_track_id_to_num(Path("./extracted-data/track/track_id_to_num.dat"))

# instantiate artist retriever
artist_retriever = ArtistRetriever()
artist_retriever.load_num_to_artist_id(Path("./extracted-data/artist/num_to_artist_id.dat"))
artist_retriever.load_artist_id_to_num(Path("./extracted-data/artist/artist_id_to_num.dat"))


# instantiate ALS using implicit
track_implict_model = implicit.als.AlternatingLeastSquares(
    factors=50, iterations=10, regularization=0.01
)
artist_implict_model = implicit.als.AlternatingLeastSquares(
    factors=50, iterations=10, regularization=0.01
)

# instantiate recommender, fit, and recommend
recommender = ImplicitRecommender(
    track_retriever=track_retriever,
    track_implicit_model=track_implict_model,
    artist_retriever=artist_retriever,
    artist_implicit_model=artist_implict_model
)

print("===FITTING TRACK MODEL===")
recommender.fit_track_model(user_songs)
print("===FITTING ARTIST MODEL===")
recommender.fit_artist_model(user_artist)



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:3000"] nếu bạn muốn giới hạn
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép POST, GET, OPTIONS, v.v.
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"Hello": "World"}




# Response: {result: [trackId1, trackId2, ...]}
@app.post(
    "/recommend-similar-tracks",
)
def read_item(schema: ReqRecommendSimilarTracks):
    
    try:
        # convert trackIds (spotify track ids - string) to songIds (our song ids)
        
        
        trackIndexNumbers =[]
        print("trackId map: ", schema.trackIds)  
        for trackId in schema.trackIds:
            try: 
                number = track_retriever.get_num_from_track_id(trackId)
                trackIndexNumbers.append(number)
            except:
                continue
        
        
        print("trackIndexNumbers: ", trackIndexNumbers)    
        
        if(trackIndexNumbers is None or len(trackIndexNumbers) == 0):
            return {
                "result": []
            }
        
        songs = recommender.recommend_similar_tracks(trackIndexNumbers, n=25)
        if songs is None:
            return {
                "result": []
            }

        # return song ids separate by comma ","
        return {
            "result": songs
        }
    except Exception as err:
        print("Error", err)
        return {
            "result": []
        }

@app.post(
    "/recommend-similar-artists",
)
def read_item(schema: ReqRecommendSimilarArtistSchema):
    
    try:
        # convert artistId (spotify artist id - string) to artistIndexNumber (our artist index number in the mapping file)
        artistIndexNumber =artist_retriever.get_num_from_artist_id(schema.artistId)
        
        if artistIndexNumber is None:
            return {
                "result": []
            }
        
        similar_artists = recommender.recommend_similar_artists(given_artist_number=artistIndexNumber, n=10)
        if similar_artists is None or len(similar_artists) == 0:
            return {
                "result": []
            }

        # return artist ids separate by comma ","
        return {
            "result": similar_artists
        }
    except: 
        print("Error")
        return {
            "result": []
        }