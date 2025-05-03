from pydantic import BaseModel


class ReqRecommendSimilarTracks(BaseModel):
    trackIds: list[str]

class ReqRecommendSimilarArtistSchema(BaseModel):
    artistId: str
