import os
import pyarrow.parquet as pq
import pandas as pd
from src.simulator import UserSimulator
from src.utils import read_yaml, read_json
from src.data.data import load_data
import asyncio
import argparse
import random

async def run_simulation(user_simulator, max_turns):
    accepted, turn = await user_simulator.interact(max_turns=max_turns)
    return accepted, turn

async def main():
    args = parse_arguments()
    recommend_multiple = args.recommend_multiple
    datasettype = args.datasettype
    path = "/data/crs/movielens/final/240911"
    movies = {
        int(k): v for k, v in read_json(os.path.join(path, "movies.json")).items()
    }
    persona_path = path + "/persona/"
    batch_size = 10
    evaluation_batch = random.sample(range(4177), batch_size)
    max_turns = 5
    HT = 0.0
    Sc = 0.0
    turns = 0

    tasks = []
    for i in evaluation_batch:
        dataset_path = persona_path + f"train_{i}.json"
        persona = read_json(dataset_path)

        target_item = movies[persona["movie_id"]]
        target_movie = target_item['title']

        if datasettype == "MACRS":
            target_movie_information = {k: v for k, v in target_item.items() if k != 'title'}
        else:
            target_movie_information = persona

        user_simulator = UserSimulator(target_movie, target_movie_information, datasettype, recommend_multiple)
        tasks.append(run_simulation(user_simulator, max_turns))

    results = await asyncio.gather(*tasks)

    for accepted, turn in results:
        if turn == max_turns:
            if accepted:
                HT += 1
        else:
            if accepted:
                Sc += 1
        turns += turn

    print("Success rate:", Sc/batch_size)
    print("Hit rate:", (HT+Sc)/batch_size)
    print("Average turn:", turns/batch_size)

def parse_arguments():
    parser = argparse.ArgumentParser(description="데이터셋 유형과 추천 옵션을 설정합니다.")
    parser.add_argument(
        "-d", "--datasettype",
        type=str,
        choices=["MACRS", "persona"],
        default="MACRS",
        help="사용할 데이터셋 유형을 선택합니다. (MACRS 또는 persona)"
    )
    parser.add_argument(
        "-r", "--recommend_multiple",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="다중 추천 여부를 설정합니다. (True 또는 False)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())