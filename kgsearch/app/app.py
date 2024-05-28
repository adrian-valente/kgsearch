import collections
import json
import os
import pathlib
import pickle
from typing import cast, Literal
from typing_extensions import get_args
from functools import lru_cache

import pandas as pd
from flask import Flask
from flask_cors import CORS, cross_origin

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from hippollm.storage import EntityStore

__all__ = ["Search", "create_app"]


# def save_metadata(origin, source):
#     """Export metadata to the library."""
#     with open(origin, "r") as f:
#         metadata = json.load(f)

#     with open(source, "w") as f:
#         json.dump(metadata, f, indent=4)

TQueryType = Literal["entity", "fact"]

class Search:
    """Search over KG."""

    def __init__(self, path: os.PathLike, emb: str) -> None:

        self.colors = ["#00A36C", "#9370DB", "#bbae98", "#7393B3", "#677179", "#318ce7", "#088F8F"]
        
        embed_model = SentenceTransformerEmbeddings(model_name=emb)
        self.storage = EntityStore(embedding_model=embed_model, persist_dir=path)
        print(f"Loaded db with {len(self.storage.entities)} entities and {len(self.storage.facts)} facts.")
        self.metadata = {}

        self.triples = collections.defaultdict(tuple)
        self.relations = collections.defaultdict(list)

    # def save(self, path):
    #     """Save the search object."""
    #     with open(path, "wb") as f:
    #         pickle.dump(self, f)
    #     return self

    # def load_metadata(self, path):
    #     """Load metadata"""
    #     with open(path, "r") as f:
    #         self.metadata = json.load(f)
    #     return self

    # @lru_cache(maxsize=10000)
    def explore(self, 
                origin: str, 
                relations: list[tuple[str, list[int], str]], 
                visited: set[str],
                depth: int, 
                max_depth: int,
                ) -> list[str]:
        depth += 1
        for neighbour, facts in self.storage.get_neighbours(origin, return_facts=True):
            relations += [tuple([origin, facts, neighbour])]
            if depth < max_depth and neighbour not in visited:
                visited.add(neighbour)
                relations = self.explore(
                    neighbour,
                    relations,
                    visited,
                    depth,
                    max_depth,
                )
        return relations

    def __call__(self, 
                 query: str, 
                 query_type: TQueryType, 
                 k: int, 
                 n: int, 
                 p: int
                 ) -> dict[str, list[dict[str, str]]]:
        nodes, links = [], []
        # prune = collections.defaultdict(int)

        for q in query.split(";"):
            q = q.strip()
            if query_type == "entity":
                entities = self.storage.get_closest_entities(q, k)
            elif query_type == "fact":
                facts = self.storage.get_closest_facts(q, k)
                entities = set()
                for fact in facts:
                    entities.add(fact.entities)
                entities = [self.storage.get_entity(e) for e in entities]

        for group, e in enumerate(entities):
            nodes.append(
                {
                    "id": e.name,
                    "group": group,
                    "color": "#960018",
                    "fontWeight": "bold",
                    "metadata": e.description,
                }
            )

        # Search for neighbours
        already_seen = {e.name for e in entities}
        added_to_plot = already_seen.copy()
        for group, e in enumerate(entities):
            color = self.colors[group % len(self.colors)]
            match = self.explore(e.name, [], already_seen, 0, n)
            for a, fcts, b in list(match):
                for x in (a, b):
                    if x not in added_to_plot:
                        x_ent = self.storage.get_entity(x)
                        nodes.append(
                            {
                                "id": x,
                                "group": group,
                                "color": color,
                                "metadata": x_ent.description,
                            }
                        )
                        added_to_plot.add(x)
                links.append(
                    {
                        "source": a,
                        "target": b,
                        "value": 1,
                        "relation": str(fcts),
                    }
                )
        # Prune
        # if p > 1:
        #     links = [
        #         link for link in links if prune[link["source"]] >= p and prune[link["target"]] >= p
        #     ]

        #     nodes = [node for node in nodes if prune[node["id"]] >= p]

        return {"nodes": nodes, "links": links}


def create_app(path: os.PathLike, emb: str) -> Flask:
    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
    app.config["CORS_HEADERS"] = "Content-Type"
    CORS(app, resources={r"/search/*": {"origins": "*"}})
    
    search = Search(path, emb)

    @app.route("/search/<k>/<n>/<p>/<query_type>/<query>", methods=["GET"])
    @cross_origin()
    def get(k: int, n: int, p: int, query: str, query_type: str):
        assert query_type in get_args(TQueryType)
        query_type = cast(TQueryType, query_type)
        return json.dumps(search(query=query, query_type=query_type, k=int(k), n=int(n), p=int(p)))

    return app
