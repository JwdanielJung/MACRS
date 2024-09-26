import os
import pyarrow.parquet as pq
import pandas as pd

from src.simulator import UserSimulator
from src.utils import read_yaml, read_json
from src.data.data import load_data
import asyncio
import argparse

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="데이터셋 유형과 추천 옵션을 설정합니다.")
        
        # datasettype 인자 추가
        parser.add_argument(
            "-d","--datasettype",
            type=str,
            choices=["MACRS", "persona"],
            default="MACRS",
            help="사용할 데이터셋 유형을 선택합니다. (MACRS 또는 persona)"
        )
        
        # recommend_multiple 인자 추가
        parser.add_argument(
            "-r","--recommend_multiple",
            type=lambda x: (str(x).lower() == 'true'),
            default=True,
            help="다중 추천 여부를 설정합니다. (True 또는 False)"
        )
        
        return parser.parse_args()
    
    args = parse_arguments()
    recommend_multiple = args.recommend_multiple
    datasettype = args.datasettype
    path = "/data/crs/movielens/final/240911"
    data = load_data(path)
    train = data["train"]
    movies = {
        int(k): v for k, v in read_json(os.path.join(path, "movies.json")).items()
    }
    
    persona_path = path + "/persona/"
    print(recommend_multiple)
    print(datasettype)
    for i, row in train.iterrows():
        dataset_path = persona_path+ f"train_{i}.json"
        persona = read_json(dataset_path)

        target_item = movies[persona["movie_id"]]
        target_movie = target_item['title']
        # break
        if datasettype == "MACRS":
            target_movie_information = {k: v for k, v in target_item.items() if k != 'title'}
        else:
            target_movie_information = persona
        
        # # Initialize user simulator
        user_simulator = UserSimulator(target_item, target_movie_information,datasettype,recommend_multiple)

        # # Start the interaction
        asyncio.run(user_simulator.interact(max_turns=5))
        break
