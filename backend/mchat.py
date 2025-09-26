# # chatbot_full_govt.py
# # Unified Mangalore Transport Chatbot (user + government)
# # - Deterministic govt answers from govt dataset (vehicle deployment, congestion, peak hours, recommendations)
# # - RAG + Gemini fallback for free-form questions
# # - Fuzzy location matching

# import os
# import random
# import re
# import difflib
# from datetime import datetime, timedelta
# from typing import Dict, List, Optional

# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Optional: Gemini (used only as fallback for free-form answers)
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# # -------------------------
# # DATA GENERATION (User trips)
# # -------------------------
# def generate_mangalore_transport_data(num_records: int = 1000) -> pd.DataFrame:
#     locations = {
#         'Pumpwell': {'rush_prob': 0.85, 'peak_hours': [7,8,9,17,18]},
#         'Hampankatta': {'rush_prob': 0.75, 'peak_hours': [9,10,17,18]},
#         'State Bank': {'rush_prob': 0.8, 'peak_hours': [8,9,17,18]},
#         'Balmatta': {'rush_prob': 0.7, 'peak_hours': [9,10,16,17]},
#         'Kadri': {'rush_prob': 0.7, 'peak_hours': [7,8,16,17]},
#         'Kottara': {'rush_prob': 0.68, 'peak_hours': [8,9,17,18]},
#         'Bejai': {'rush_prob': 0.65, 'peak_hours': [7,8,16,17]},
#         'Lalbagh': {'rush_prob': 0.6, 'peak_hours': [8,9,18,19]},
#         'Kankanady': {'rush_prob': 0.55, 'peak_hours': [7,8,18,19]},
#         'Bendoorwell': {'rush_prob': 0.5, 'peak_hours': [8,9,17,18]},
#         'Falnir': {'rush_prob': 0.52, 'peak_hours': [8,18]},
#         'Valencia': {'rush_prob': 0.45, 'peak_hours': [8,18]},
#         'Urva': {'rush_prob': 0.48, 'peak_hours': [8,9,17]},
#         'Deralakatte': {'rush_prob': 0.4, 'peak_hours': [8,17]},
#         'Kulai': {'rush_prob': 0.35, 'peak_hours': [8,17]},
#         'Mangaladevi': {'rush_prob': 0.65, 'peak_hours': [7,8,18,19]},
#         'Jyothi': {'rush_prob': 0.6, 'peak_hours': [8,9,17]},
#         'Attavar': {'rush_prob': 0.55, 'peak_hours': [8,18]},
#         'Kodialbail': {'rush_prob': 0.6, 'peak_hours': [8,9,17,18]},
#         'Shivbagh': {'rush_prob': 0.5, 'peak_hours': [8,18]},
#         'Ballalbagh': {'rush_prob': 0.58, 'peak_hours': [8,9,17]},
#         'Pandeshwar': {'rush_prob': 0.72, 'peak_hours': [7,8,17,18]},
#         'Car Street': {'rush_prob': 0.8, 'peak_hours': [10,11,17,18,19]},
#         'Milagres': {'rush_prob': 0.6, 'peak_hours': [8,9,16,17]},
#         'Bolar': {'rush_prob': 0.5, 'peak_hours': [8,9,17]},
#         'PVS Circle': {'rush_prob': 0.55, 'peak_hours': [8,9,17]},
#         'Light House Hill': {'rush_prob': 0.6, 'peak_hours': [9,10,17]},
#         'Old Port': {'rush_prob': 0.45, 'peak_hours': [8,17]}
#     }

#     transport_modes = ['Bus', 'Auto', 'Car', 'Two-Wheeler', 'Walk', 'Taxi']
#     data = []

#     for i in range(num_records):
#         origin = random.choice(list(locations.keys()))
#         destination = random.choice([l for l in locations.keys() if l != origin])
#         location_peaks = locations[origin]['peak_hours'] + locations[destination]['peak_hours']

#         # hour weighting
#         hour_weights = [0.3] * 24
#         for h in location_peaks:
#             hour_weights[h] += 1.5
#         for h in [7,8,9,17,18,19]:
#             hour_weights[h] += 0.5
#         hour = random.choices(range(24), weights=hour_weights)[0]
#         minute = random.randint(0,59)
#         day = random.randint(1,28)
#         start_time = datetime(2025, 1, day, hour, minute)
#         duration = random.randint(10,90)
#         end_time = start_time + timedelta(minutes=duration)
#         distance = round(random.uniform(1.5, 25.0), 2)
#         rush_factor = (locations[origin]['rush_prob'] + locations[destination]['rush_prob']) / 2
#         if 7 <= hour <= 9 or 17 <= hour <= 19:
#             rush_factor += 0.3
#         if rush_factor > 0.7:
#             rush_level = 'High'
#         elif rush_factor > 0.4:
#             rush_level = 'Medium'
#         else:
#             rush_level = 'Low'
#         if rush_level == 'High':
#             mode = random.choices(transport_modes, weights=[3,4,2,5,1,2])[0]
#         else:
#             mode = random.choices(transport_modes, weights=[2,3,3,3,2,1])[0]
#         data.append({
#             'TripID': i+1,
#             'UserID': random.randint(1, 200),
#             'Origin': origin,
#             'Destination': destination,
#             'StartTime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
#             'EndTime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
#             'Mode': mode,
#             'RushLevel': rush_level,
#             'DistanceKM': distance,
#             'Hour': hour
#         })

#     # ensure each location appears at least once
#     for loc in locations.keys():
#         if not any(d['Origin'] == loc or d['Destination'] == loc for d in data):
#             hour = random.randint(6,20)
#             st = datetime(2025,1,random.randint(1,28),hour,random.randint(0,59))
#             data.append({
#                 'TripID': len(data)+1,
#                 'UserID': random.randint(1,200),
#                 'Origin': loc,
#                 'Destination': random.choice([l for l in locations.keys() if l != loc]),
#                 'StartTime': st.strftime('%Y-%m-%d %H:%M:%S'),
#                 'EndTime': (st + timedelta(minutes=random.randint(10,60))).strftime('%Y-%m-%d %H:%M:%S'),
#                 'Mode': random.choice(transport_modes),
#                 'RushLevel': random.choice(['Low','Medium','High']),
#                 'DistanceKM': round(random.uniform(1.5,25.0),2),
#                 'Hour': hour
#             })

#     return pd.DataFrame(data)

# # -------------------------
# # GOVERNMENT DATASET GENERATION
# # -------------------------
# def generate_govt_transport_data(locations_info: Dict[str, Dict]) -> pd.DataFrame:
#     """
#     Build a govt dataset for each location with suggested vehicles, congestion index,
#     weekend/market/festival adjustments, parking & bus-stop recommendations.
#     """
#     rows = []
#     for loc, info in locations_info.items():
#         congestion_index = round(min(1.0, max(0.0, random.gauss(info.get('rush_prob', 0.5), 0.12))), 2)
#         avg_daily_trips = random.randint(100, 1500)  # simulated
#         suggested_buses = max(0, int(avg_daily_trips * congestion_index / 200))  # heuristic
#         suggested_autos = max(0, int(avg_daily_trips * congestion_index / 300))
#         suggested_taxis = max(0, int(avg_daily_trips * congestion_index / 500))
#         weekend_adj = random.choice(['Increase', 'Decrease', 'No Change'])
#         festival_adj = 'Increase' if info.get('type','') == 'temple_area' else random.choice(['No Change','Increase'])
#         parking_req = 'Yes' if congestion_index > 0.6 and info.get('type') in ['commercial', 'major_junction'] else 'No'
#         busstop_rec = 'Add' if congestion_index > 0.5 else 'No Change'

#         rows.append({
#             'Location': loc,
#             'PeakHours': info.get('peak_hours', []),
#             'AvgDailyTrips': avg_daily_trips,
#             'CongestionIndex': congestion_index,
#             'RushLevel': 'High' if congestion_index > 0.7 else ('Medium' if congestion_index > 0.4 else 'Low'),
#             'SuggestedBuses': suggested_buses,
#             'SuggestedAutos': suggested_autos,
#             'SuggestedTaxis': suggested_taxis,
#             'WeekendAdjustment': weekend_adj,
#             'FestivalAdjustment': festival_adj,
#             'ParkingRequired': parking_req,
#             'BusStopRecommendation': busstop_rec,
#             'PopularMode': random.choice(['Bus','Auto','Two-Wheeler','Car'])
#         })
#     return pd.DataFrame(rows)

# # -------------------------
# # Fuzzy location helper
# # -------------------------
# def fuzzy_match_location(query: str, location_list: List[str]) -> Optional[str]:
#     q = query.lower()
#     # first try exact containment
#     for loc in location_list:
#         if loc.lower() in q:
#             return loc
#     # try close match against words
#     tokens = re.findall(r"[A-Za-z ]+", query)
#     combined = ' '.join(tokens).strip().lower()
#     matches = difflib.get_close_matches(combined, [l.lower() for l in location_list], n=1, cutoff=0.6)
#     if matches:
#         matched = matches[0]
#         for loc in location_list:
#             if loc.lower() == matched:
#                 return loc
#     # token-wise tries
#     for token in combined.split():
#         m = difflib.get_close_matches(token, [l.lower() for l in location_list], n=1, cutoff=0.7)
#         if m:
#             for loc in location_list:
#                 if loc.lower() == m[0]:
#                     return loc
#     return None

# # -------------------------
# # RAG system (user + govt)
# # -------------------------
# class CombinedRAG:
#     def __init__(self, gemini_api_key: Optional[str], user_df: pd.DataFrame, govt_df: pd.DataFrame):
#         # configure gemini (optional)
#         self.gemini_key = gemini_api_key
#         if gemini_api_key:
#             try:
#                 genai.configure(api_key=gemini_api_key)
#                 # don't raise here if model not available; we'll still answer from data
#                 self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
#             except Exception:
#                 self.gemini_model = None
#         else:
#             self.gemini_model = None

#         self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
#         self.knowledge_base = []
#         self.user_df = user_df.copy()
#         self.govt_df = govt_df.copy()
#         self._build_knowledge_base(user_df, govt_df)

#     def _build_knowledge_base(self, user_df: pd.DataFrame, govt_df: pd.DataFrame):
#         self.knowledge_base = []
#         # user trips -> concise sentences
#         for _, row in user_df.iterrows():
#             doc = f"Trip from {row['Origin']} to {row['Destination']} using {row['Mode']}. Rush level: {row['RushLevel']}. Distance: {row['DistanceKM']} km. Time: {row['StartTime']}."
#             self.knowledge_base.append({'content': doc.lower(), 'metadata': row.to_dict()})
#         # govt rows -> descriptive
#         for _, row in govt_df.iterrows():
#             doc = (f"Government data for {row['Location']}: Peak hours {row['PeakHours']}. Congestion index: {row['CongestionIndex']}, "
#                    f"rush level: {row['RushLevel']}. Suggested buses: {row['SuggestedBuses']}, autos: {row['SuggestedAutos']}, taxis: {row['SuggestedTaxis']}. "
#                    f"Weekend adjustment: {row['WeekendAdjustment']}. Festival adjustment: {row['FestivalAdjustment']}. Parking required: {row['ParkingRequired']}. "
#                    f"Bus stop recommendation: {row['BusStopRecommendation']}. Popular mode: {row['PopularMode']}.")
#             self.knowledge_base.append({'content': doc.lower(), 'metadata': row.to_dict()})
#         documents = [d['content'] for d in self.knowledge_base]
#         if documents:
#             self.document_vectors = self.vectorizer.fit_transform(documents)
#         else:
#             self.document_vectors = None

#     def _retrieve(self, query: str, top_k: int = 6):
#         if not self.document_vectors is None:
#             qv = self.vectorizer.transform([query.lower()])
#             sims = cosine_similarity(qv, self.document_vectors).flatten()
#             idx = np.argsort(sims)[::-1][:top_k]
#             results = []
#             for i in idx:
#                 if sims[i] > 0.08:
#                     results.append({'content': self.knowledge_base[i]['content'], 'metadata': self.knowledge_base[i]['metadata'], 'sim': float(sims[i])})
#             return results
#         return []

#     # ------------ DETERMINISTIC GOVT HANDLERS ------------
#     def gov_how_many_buses(self, location: str) -> str:
#         row = self.govt_df[self.govt_df['Location'].str.lower() == location.lower()]
#         if row.empty:
#             return f"No government planning data available for {location}."
#         r = row.iloc[0]
#         return (f"Suggested buses for {r['Location']}: {int(r['SuggestedBuses'])}. "
#                 f"Peak hours: {', '.join(str(h) + ':00' for h in r['PeakHours']) if r['PeakHours'] else 'Not listed'}. "
#                 f"Congestion: {r['RushLevel']}. Weekend adj: {r['WeekendAdjustment']}. Bus stop recommendation: {r['BusStopRecommendation']}.")

#     def gov_area_summary(self, location: str) -> str:
#         row = self.govt_df[self.govt_df['Location'].str.lower() == location.lower()]
#         if row.empty:
#             return f"No government planning data available for {location}."
#         r = row.iloc[0]
#         return (f"{r['Location']}: Rush {r['RushLevel']}, CongestionIndex {r['CongestionIndex']}. "
#                 f"Suggested - Buses: {r['SuggestedBuses']}, Autos: {r['SuggestedAutos']}, Taxis: {r['SuggestedTaxis']}. "
#                 f"Popular mode: {r['PopularMode']}. Parking required: {r['ParkingRequired']}. Bus stop: {r['BusStopRecommendation']}.")

#     def gov_all_places_summary(self) -> str:
#         parts = []
#         for _, r in self.govt_df.iterrows():
#             parts.append(f"{r['Location']}: Buses {int(r['SuggestedBuses'])}, Rush {r['RushLevel']}, Popular {r['PopularMode']}")
#         return "\n".join(parts)

#     def gov_top_congested(self, top_n: int = 5) -> str:
#         df = self.govt_df.sort_values('CongestionIndex', ascending=False).head(top_n)
#         return ", ".join(f"{row['Location']}({row['CongestionIndex']})" for _, row in df.iterrows())

#     # main public interface
#     def answer(self, query: str) -> str:
#         q = query.strip()
#         q_lower = q.lower()

#         # Check for "all places" summary request
#         if any(tok in q_lower for tok in ["all places", "all areas", "all locations", "summary for all"]):
#             return self.gov_all_places_summary()

#         # detect location (fuzzy)
#         loc = fuzzy_match_location(q, list(self.govt_df['Location'].unique()))

#         # GOVERNMENT INTENT KEYWORDS
#         gov_keywords = [
#             "how many buses", "how many bus", "suggested buses", "deploy buses", "deploy bus", "deploy autos", "public transport",
#             "which areas need more buses", "more buses", "should we increase auto", "increase auto", "taxis", "taxi", "suggested taxis",
#             "parking", "bus stop", "bus-stop", "bus stop recommendation", "congestion", "which areas have the highest", "peak hours",
#             "festival", "weekend", "market", "school timings", "safe routes", "add bus stops", "parking facilities"
#         ]
#         if any(k in q_lower for k in gov_keywords):
#             # If asking "how many buses" explicitly and location found
#             if any(k in q_lower for k in ["how many buses", "how many bus", "suggested buses", "deploy buses", "deploy bus"]) and loc:
#                 return self.gov_how_many_buses(loc)
#             # "which areas need more buses during morning/evening peaks"
#             if "which areas" in q_lower and "buses" in q_lower:
#                 # choose top congestion indices where suggested buses > 0
#                 df = self.govt_df.copy()
#                 df['need_more_buses_score'] = df['CongestionIndex'] * (df['SuggestedBuses'] + 1)
#                 top = df.sort_values('need_more_buses_score', ascending=False).head(5)
#                 return "Areas that need more buses (priority): " + ", ".join(f"{r['Location']} (suggested {int(r['SuggestedBuses'])})" for _, r in top.iterrows())
#             # Increase auto coverage in a location
#             if "auto" in q_lower and ("increase" in q_lower or "coverage" in q_lower or "should we" in q_lower):
#                 if loc:
#                     r = self.govt_df[self.govt_df['Location'].str.lower() == loc.lower()].iloc[0]
#                     return (f"For {r['Location']}: suggested autos = {int(r['SuggestedAutos'])}. Congestion: {r['RushLevel']}. "
#                             f"Weekend adj: {r['WeekendAdjustment']}.")
#                 else:
#                     return "Specify the location (e.g., 'Should we increase auto coverage in Kadri?')."
#             # taxis in Car Street during weekends
#             if "taxi" in q_lower and "weekend" in q_lower and loc:
#                 r = self.govt_df[self.govt_df['Location'].str.lower() == loc.lower()].iloc[0]
#                 return (f"For {r['Location']} on weekends: suggested taxis = {int(r['SuggestedTaxis'])}. Weekend adjustment: {r['WeekendAdjustment']}.")
#             # congestion analysis queries
#             if "which areas have the highest" in q_lower or "highest congestion" in q_lower or "most congested" in q_lower:
#                 top = self.govt_top_congested(5)
#                 return f"Top congested areas: {top}"
#             if "peak hours in" in q_lower or ("what are the peak hours" in q_lower):
#                 if loc:
#                     r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
#                     ph = r['PeakHours']
#                     return f"Peak hours for {r['Location']}: {', '.join(str(h)+':00' for h in ph)}"
#                 else:
#                     return "Specify the location for peak hours (e.g., 'What are the peak hours in Balmatta?')."
#             if "compare" in q_lower and loc:
#                 # compare to another location
#                 other_loc = None
#                 for candidate in self.govt_df['Location'].unique():
#                     if candidate.lower() in q_lower and candidate.lower() != loc.lower():
#                         other_loc = candidate
#                         break
#                 if other_loc:
#                     r1 = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
#                     r2 = self.govt_df[self.govt_df['Location'].str.lower()==other_loc.lower()].iloc[0]
#                     return (f"{r1['Location']} - Rush: {r1['RushLevel']}, Congestion: {r1['CongestionIndex']}. "
#                             f"{r2['Location']} - Rush: {r2['RushLevel']}, Congestion: {r2['CongestionIndex']}.")
#             # rush hour / signal schedule
#             if "schedule traffic signals" in q_lower or "traffic signals" in q_lower:
#                 if loc:
#                     r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
#                     ph = r['PeakHours']
#                     return f"Schedule signal prioritization for {r['Location']} during peak windows: {', '.join(str(h)+':00' for h in ph)}"
#             # event/weekend/market planning
#             if "weekend" in q_lower or "festival" in q_lower or "market" in q_lower:
#                 if loc:
#                     r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
#                     return (f"{r['Location']}: Weekend adj = {r['WeekendAdjustment']}, Festival adj = {r['FestivalAdjustment']}. Suggested buses: {int(r['SuggestedBuses'])}.")
#             # parking / bus stop / bus stop recommendation
#             if any(k in q_lower for k in ["parking", "parking facilities", "add parking"]):
#                 if loc:
#                     r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
#                     return f"{r['Location']}: Parking required? {r['ParkingRequired']}. Congestion: {r['RushLevel']}."
#             if any(k in q_lower for k in ["bus stop", "bus-stop", "add bus stop"]):
#                 if loc:
#                     r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
#                     return f"{r['Location']}: Bus stop recommendation: {r['BusStopRecommendation']}."
#             # safe routes / school timings / least congested
#             if "safe routes" in q_lower or "school" in q_lower:
#                 # use govt 'PopularMode' & user_df stats to suggest routes — keep simple
#                 candidates = self.govt_df.sort_values('CongestionIndex').head(5)
#                 return "Less congested areas (possible safe routes): " + ", ".join(candidates['Location'].tolist()[:5])

#         # If not a clear govt intent, try standard retrieval + Gemini fallback
#         docs = self._retrieve(q, top_k=6)
#         if docs:
#             context = "\n".join(d['content'] for d in docs)
#             # If Gemini available, ask it to generate a concise answer using context
#             if self.gemini_model:
#                 prompt = f"You are a Mangalore transport expert. Use the context below to answer concisely (1-2 sentences):\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
#                 try:
#                     resp = self.gemini_model.generate_content(prompt)
#                     raw = resp.text if hasattr(resp, 'text') else str(resp)
#                     # keep it short
#                     return raw.strip()
#                 except Exception:
#                     pass
#             # fallback: return a concise synthesis from retrieved docs
#             # choose the top doc and return a short sentence
#             top_doc = docs[0]['content']
#             # craft a short answer from the doc
#             short = re.split(r'(?<=[.!?])\s+', top_doc)
#             if short:
#                 answer = short[0].capitalize()
#                 if not answer.endswith('.'):
#                     answer += '.'
#                 return answer
#         return "I don't have enough information to answer that question about Mangalore transportation."

# # -------------------------
# # CLI Chatbot wrapper
# # -------------------------
# class MangaloreTransportChatbot:
#     def __init__(self):
#         print("Initializing Mangalore Transport Chatbot (user + govt)...")
#         self.user_data = generate_mangalore_transport_data(1200)
#         # build locations info for govt dataset
#         locs_info = {}
#         # derive locations list from user_data
#         for loc in sorted(set(self.user_data['Origin'].tolist() + self.user_data['Destination'].tolist())):
#             # estimate rush_prob from user data's rush level mapping
#             subset = self.user_data[(self.user_data['Origin'] == loc) | (self.user_data['Destination'] == loc)]
#             if not subset.empty:
#                 mapping = {'Low': 0.2, 'Medium': 0.55, 'High': 0.9}
#                 avg_rush = subset['RushLevel'].map(mapping).mean()
#                 peak_hours = subset['Hour'].value_counts().head(3).index.tolist()
#             else:
#                 avg_rush = 0.5
#                 peak_hours = [8,17]
#             locs_info[loc] = {'rush_prob': avg_rush, 'peak_hours': peak_hours, 'type': 'mixed'}
#         # generate govt dataset
#         self.govt_data = generate_govt_transport_data(locs_info)
#         # configure gemini key (optional)
#         gemini_key = os.getenv("GEMINI_API_KEY")
#         self.rag = CombinedRAG(gemini_key, self.user_data, self.govt_data)
#         print("Chatbot ready. Ask user or government questions. Type 'quit' to exit.")

#     def chat(self):
#         while True:
#             try:
#                 q = input("\n💬 You: ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 print("\n👋 Bye!")
#                 break
#             if not q:
#                 continue
#             if q.lower() in ["quit", "exit", "bye"]:
#                 print("👋 Bye!")
#                 break
#             resp = self.rag.answer(q)
#             print("\n🤖 Bot:", resp)

# # -------------------------
# # MAIN
# # -------------------------
# def main():
#     bot = MangaloreTransportChatbot()
#     bot.chat()

# if __name__ == "__main__":
#     main()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import uvicorn
import random
import re
import difflib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import google.generativeai as genai
except Exception:
    genai = None
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()


def generate_mangalore_transport_data(num_records: int = 1000) -> pd.DataFrame:
    locations = {
        # (same as your pasted code...)
        'Pumpwell': {'rush_prob': 0.85, 'peak_hours': [7,8,9,17,18]},
        # ... all locations ...
        'Old Port': {'rush_prob': 0.45, 'peak_hours': [8,17]}
    }
    transport_modes = ['Bus', 'Auto', 'Car', 'Two-Wheeler', 'Walk', 'Taxi']
    data = []
    for i in range(num_records):
        origin = random.choice(list(locations.keys()))
        destination = random.choice([l for l in locations.keys() if l != origin])
        location_peaks = locations[origin]['peak_hours'] + locations[destination]['peak_hours']
        hour_weights = [0.3]*24
        for h in location_peaks:
            hour_weights[h] += 1.5
        for h in [7,8,9,17,18,19]:
            hour_weights[h] += 0.5
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0,59)
        day = random.randint(1,28)
        start_time = datetime(2025,1,day,hour,minute)
        duration = random.randint(10,90)
        end_time = start_time + timedelta(minutes=duration)
        distance = round(random.uniform(1.5,25.0),2)
        rush_factor = (locations[origin]['rush_prob'] + locations[destination]['rush_prob']) / 2
        if 7 <= hour <=9 or 17 <= hour <=19:
            rush_factor += 0.3
        rush_level = 'High' if rush_factor > 0.7 else ('Medium' if rush_factor > 0.4 else 'Low')
        if rush_level == 'High':
            mode = random.choices(transport_modes, weights=[3,4,2,5,1,2])[0]
        else:
            mode = random.choices(transport_modes, weights=[2,3,3,3,2,1])[0]
        data.append({
            'TripID': i+1,
            'UserID': random.randint(1,200),
            'Origin': origin,
            'Destination': destination,
            'StartTime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EndTime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Mode': mode,
            'RushLevel': rush_level,
            'DistanceKM': distance,
            'Hour': hour
        })
    for loc in locations.keys():
        if not any(d['Origin'] == loc or d['Destination'] == loc for d in data):
            hour = random.randint(6,20)
            st = datetime(2025,1,random.randint(1,28),hour,random.randint(0,59))
            data.append({
                'TripID': len(data)+1,
                'UserID': random.randint(1,200),
                'Origin': loc,
                'Destination': random.choice([l for l in locations.keys() if l != loc]),
                'StartTime': st.strftime('%Y-%m-%d %H:%M:%S'),
                'EndTime': (st + timedelta(minutes=random.randint(10,60))).strftime('%Y-%m-%d %H:%M:%S'),
                'Mode': random.choice(transport_modes),
                'RushLevel': random.choice(['Low','Medium','High']),
                'DistanceKM': round(random.uniform(1.5,25.0),2),
                'Hour': hour
            })
    return pd.DataFrame(data)

def generate_govt_transport_data(locations_info: dict) -> pd.DataFrame:
    rows = []
    for loc, info in locations_info.items():
        congestion_index = round(min(1.0, max(0.0, random.gauss(info.get('rush_prob', 0.5), 0.12))),2)
        avg_daily_trips = random.randint(100,1500)
        suggested_buses = max(0, int(avg_daily_trips*congestion_index/200))
        suggested_autos = max(0, int(avg_daily_trips*congestion_index/300))
        suggested_taxis = max(0, int(avg_daily_trips*congestion_index/500))
        weekend_adj = random.choice(['Increase','Decrease','No Change'])
        festival_adj = 'Increase' if info.get('type', '') == 'temple_area' else random.choice(['No Change','Increase'])
        parking_req = 'Yes' if congestion_index > 0.6 and info.get('type') in ['commercial', 'major_junction'] else 'No'
        busstop_rec = 'Add' if congestion_index > 0.5 else 'No Change'
        rows.append({
            'Location': loc,
            'PeakHours': info.get('peak_hours', []),
            'AvgDailyTrips': avg_daily_trips,
            'CongestionIndex': congestion_index,
            'RushLevel': 'High' if congestion_index > 0.7 else ('Medium' if congestion_index > 0.4 else 'Low'),
            'SuggestedBuses': suggested_buses,
            'SuggestedAutos': suggested_autos,
            'SuggestedTaxis': suggested_taxis,
            'WeekendAdjustment': weekend_adj,
            'FestivalAdjustment': festival_adj,
            'ParkingRequired': parking_req,
            'BusStopRecommendation': busstop_rec,
            'PopularMode': random.choice(['Bus','Auto','Two-Wheeler','Car'])
        })
    return pd.DataFrame(rows)

def fuzzy_match_location(query: str, location_list: List[str]) -> Optional[str]:
    q = query.lower()
    for loc in location_list:
        if loc.lower() in q:
            return loc
    tokens = re.findall(r"[A-Za-z ]+", query)
    combined = ' '.join(tokens).strip().lower()
    matches = difflib.get_close_matches(combined, [l.lower() for l in location_list], n=1, cutoff=0.6)
    if matches:
        matched = matches[0]
        for loc in location_list:
            if loc.lower() == matched:
                return loc
    for token in combined.split():
        m = difflib.get_close_matches(token, [l.lower() for l in location_list], n=1, cutoff=0.7)
        if m:
            for loc in location_list:
                if loc.lower() == m[0]:
                    return loc
    return None

class CombinedRAG:
    def __init__(self, gemini_api_key: Optional[str], user_df: pd.DataFrame, govt_df: pd.DataFrame):
        self.gemini_key = gemini_api_key
        self.gemini_model = None
        if genai and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                self.gemini_model = None

        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.knowledge_base = []
        self.user_df = user_df.copy()
        self.govt_df = govt_df.copy()
        self._build_knowledge_base(user_df, govt_df)

    def _build_knowledge_base(self, user_df: pd.DataFrame, govt_df: pd.DataFrame):
        self.knowledge_base = []
        for _, row in user_df.iterrows():
            doc = f"Trip from {row['Origin']} to {row['Destination']} using {row['Mode']}. Rush level: {row['RushLevel']}. Distance: {row['DistanceKM']} km. Time: {row['StartTime']}."
            self.knowledge_base.append({'content': doc.lower(), 'metadata': row.to_dict()})
        for _, row in govt_df.iterrows():
            doc = (f"Government data for {row['Location']}: Peak hours {row['PeakHours']}. Congestion index: {row['CongestionIndex']}, "
                   f"rush level: {row['RushLevel']}. Suggested buses: {row['SuggestedBuses']}, autos: {row['SuggestedAutos']}, taxis: {row['SuggestedTaxis']}. "
                   f"Weekend adjustment: {row['WeekendAdjustment']}. Festival adjustment: {row['FestivalAdjustment']}. Parking required: {row['ParkingRequired']}. "
                   f"Bus stop recommendation: {row['BusStopRecommendation']}. Popular mode: {row['PopularMode']}.")
            self.knowledge_base.append({'content': doc.lower(), 'metadata': row.to_dict()})
        documents = [d['content'] for d in self.knowledge_base]
        if documents:
            self.document_vectors = self.vectorizer.fit_transform(documents)
        else:
            self.document_vectors = None

    def _retrieve(self, query: str, top_k: int = 6):
        if self.document_vectors is not None:
            qv = self.vectorizer.transform([query.lower()])
            sims = cosine_similarity(qv, self.document_vectors).flatten()
            idx = np.argsort(sims)[::-1][:top_k]
            results = []
            for i in idx:
                if sims[i] > 0.08:
                    results.append({'content': self.knowledge_base[i]['content'], 'metadata': self.knowledge_base[i]['metadata'], 'sim': float(sims[i])})
            return results
        return []

    def gov_how_many_buses(self, location: str) -> str:
        row = self.govt_df[self.govt_df['Location'].str.lower() == location.lower()]
        if row.empty:
            return f"No government planning data available for {location}."
        r = row.iloc[0]
        return (f"Suggested buses for {r['Location']}: {int(r['SuggestedBuses'])}. "
                f"Peak hours: {', '.join(str(h) + ':00' for h in r['PeakHours']) if r['PeakHours'] else 'Not listed'}. "
                f"Congestion: {r['RushLevel']}. Weekend adj: {r['WeekendAdjustment']}. Bus stop recommendation: {r['BusStopRecommendation']}.")

    def gov_area_summary(self, location: str) -> str:
        row = self.govt_df[self.govt_df['Location'].str.lower() == location.lower()]
        if row.empty:
            return f"No government planning data available for {location}."
        r = row.iloc[0]
        return (f"{r['Location']}: Rush {r['RushLevel']}, CongestionIndex {r['CongestionIndex']}. "
                f"Suggested - Buses: {r['SuggestedBuses']}, Autos: {r['SuggestedAutos']}, Taxis: {r['SuggestedTaxis']}. "
                f"Popular mode: {r['PopularMode']}. Parking required: {r['ParkingRequired']}. Bus stop: {r['BusStopRecommendation']}.")

    def gov_all_places_summary(self) -> str:
        parts = []
        for _, r in self.govt_df.iterrows():
            parts.append(f"{r['Location']}: Buses {int(r['SuggestedBuses'])}, Rush {r['RushLevel']}, Popular {r['PopularMode']}")
        return "\n".join(parts)

    def gov_top_congested(self, top_n: int = 5) -> str:
        df = self.govt_df.sort_values('CongestionIndex', ascending=False).head(top_n)
        return ", ".join(f"{row['Location']}({row['CongestionIndex']})" for _, row in df.iterrows())

    def answer(self, query: str) -> str:
        q = query.strip()
        q_lower = q.lower()
        if any(tok in q_lower for tok in ["all places", "all areas", "all locations", "summary for all"]):
            return self.gov_all_places_summary()
        loc = fuzzy_match_location(q, list(self.govt_df['Location'].unique()))
        gov_keywords = [
            "how many buses", "how many bus", "suggested buses", "deploy buses", "deploy bus", "deploy autos", "public transport",
            "which areas need more buses", "more buses", "should we increase auto", "increase auto", "taxis", "taxi", "suggested taxis",
            "parking", "bus stop", "bus-stop", "bus stop recommendation", "congestion", "which areas have the highest", "peak hours",
            "festival", "weekend", "market", "school timings", "safe routes", "add bus stops", "parking facilities"
        ]
        if any(k in q_lower for k in gov_keywords):
            if any(k in q_lower for k in ["how many buses", "how many bus", "suggested buses", "deploy buses", "deploy bus"]) and loc:
                return self.gov_how_many_buses(loc)
            if "which areas" in q_lower and "buses" in q_lower:
                df = self.govt_df.copy()
                df['need_more_buses_score'] = df['CongestionIndex'] * (df['SuggestedBuses'] + 1)
                top = df.sort_values('need_more_buses_score', ascending=False).head(5)
                return "Areas that need more buses (priority): " + ", ".join(f"{r['Location']} (suggested {int(r['SuggestedBuses'])})" for _, r in top.iterrows())
            if "auto" in q_lower and ("increase" in q_lower or "coverage" in q_lower or "should we" in q_lower):
                if loc:
                    r = self.govt_df[self.govt_df['Location'].str.lower() == loc.lower()].iloc[0]
                    return (f"For {r['Location']}: suggested autos = {int(r['SuggestedAutos'])}. Congestion: {r['RushLevel']}. "
                            f"Weekend adj: {r['WeekendAdjustment']}.")
                else:
                    return "Specify the location (e.g., 'Should we increase auto coverage in Kadri?')."
            if "taxi" in q_lower and "weekend" in q_lower and loc:
                r = self.govt_df[self.govt_df['Location'].str.lower() == loc.lower()].iloc[0]
                return (f"For {r['Location']} on weekends: suggested taxis = {int(r['SuggestedTaxis'])}. Weekend adjustment: {r['WeekendAdjustment']}.")
            if "which areas have the highest" in q_lower or "highest congestion" in q_lower or "most congested" in q_lower:
                top = self.gov_top_congested(5)
                return f"Top congested areas: {top}"
            if "peak hours in" in q_lower or ("what are the peak hours" in q_lower):
                if loc:
                    r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
                    ph = r['PeakHours']
                    return f"Peak hours for {r['Location']}: {', '.join(str(h)+':00' for h in ph)}"
                else:
                    return "Specify the location for peak hours (e.g., 'What are the peak hours in Balmatta?')."
            if "compare" in q_lower and loc:
                other_loc = None
                for candidate in self.govt_df['Location'].unique():
                    if candidate.lower() in q_lower and candidate.lower() != loc.lower():
                        other_loc = candidate
                        break
                if other_loc:
                    r1 = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
                    r2 = self.govt_df[self.govt_df['Location'].str.lower()==other_loc.lower()].iloc[0]
                    return (f"{r1['Location']} - Rush: {r1['RushLevel']}, Congestion: {r1['CongestionIndex']}. "
                            f"{r2['Location']} - Rush: {r2['RushLevel']}, Congestion: {r2['CongestionIndex']}.")
            if "schedule traffic signals" in q_lower or "traffic signals" in q_lower:
                if loc:
                    r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
                    ph = r['PeakHours']
                    return f"Schedule signal prioritization for {r['Location']} during peak windows: {', '.join(str(h)+':00' for h in ph)}"
            if "weekend" in q_lower or "festival" in q_lower or "market" in q_lower:
                if loc:
                    r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
                    return (f"{r['Location']}: Weekend adj = {r['WeekendAdjustment']}, Festival adj = {r['FestivalAdjustment']}. Suggested buses: {int(r['SuggestedBuses'])}.")
            if any(k in q_lower for k in ["parking", "parking facilities", "add parking"]):
                if loc:
                    r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
                    return f"{r['Location']}: Parking required? {r['ParkingRequired']}. Congestion: {r['RushLevel']}."
            if any(k in q_lower for k in ["bus stop", "bus-stop", "add bus stop"]):
                if loc:
                    r = self.govt_df[self.govt_df['Location'].str.lower()==loc.lower()].iloc[0]
                    return f"{r['Location']}: Bus stop recommendation: {r['BusStopRecommendation']}."
            if "safe routes" in q_lower or "school" in q_lower:
                candidates = self.govt_df.sort_values('CongestionIndex').head(5)
                return "Less congested areas (possible safe routes): " + ", ".join(candidates['Location'].tolist()[:5])
        docs = self._retrieve(q, top_k=6)
        if docs:
            context = "\n".join(d['content'] for d in docs)
            if self.gemini_model:
                prompt = f"You are a Mangalore transport expert. Use the context below to answer concisely (1-2 sentences):\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
                try:
                    resp = self.gemini_model.generate_content(prompt)
                    raw = resp.text if hasattr(resp, 'text') else str(resp)
                    return raw.strip()
                except Exception:
                    pass
            top_doc = docs[0]['content']
            short = re.split(r'(?<=[.!?])\s+', top_doc)
            if short:
                answer = short[0].capitalize()
                if not answer.endswith('.'):
                    answer += '.'
                return answer
        return "I don't have enough information to answer that question about Kerala transportation."


class ChatRequest(BaseModel):
    query: str

app = FastAPI(title="Mangalore Transport Chatbot API")

origins = ["*"]

app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)
@app.on_event("startup")
async def startup_event():
    user_data = generate_mangalore_transport_data(1200)
    locs_info = {}
    for loc in sorted(set(user_data['Origin'].tolist() + user_data['Destination'].tolist())):
        subset = user_data[(user_data['Origin'] == loc) | (user_data['Destination'] == loc)]
        if not subset.empty:
            mapping = {'Low': 0.2, 'Medium': 0.55, 'High': 0.9}
            avg_rush = subset['RushLevel'].map(mapping).mean()
            peak_hours = subset['Hour'].value_counts().head(3).index.tolist()
        else:
            avg_rush = 0.5
            peak_hours = [8,17]
        locs_info[loc] = {'rush_prob': avg_rush, 'peak_hours': peak_hours, 'type': 'mixed'}
    govt_data = generate_govt_transport_data(locs_info)
    gemini_key = os.getenv("GEMINI_API_KEY")
    app.state.rag = CombinedRAG(gemini_key, user_data, govt_data)
    app.state.user_data = user_data
    app.state.govt_data = govt_data

@app.post("/chat")
async def chat(request: ChatRequest):
    q = request.query
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty")
    rag: CombinedRAG = app.state.rag
    ans = rag.answer(q)
    return {"query": q, "answer": ans}

@app.get("/locations")
async def list_locations() -> List[str]:
    return sorted(app.state.govt_data['Location'].tolist())

@app.get("/gov/summary")
async def gov_summary(location: Optional[str] = None):
    rag: CombinedRAG = app.state.rag
    if not location:
        return {"summary": rag.gov_all_places_summary()}
    loc_match = fuzzy_match_location(location, list(app.state.govt_data['Location'].unique()))
    if not loc_match:
        raise HTTPException(status_code=404, detail="Location not found")
    return {"location": loc_match, "summary": rag.gov_area_summary(loc_match)}

@app.get("/gov/top_congested")
async def gov_top_congested(top_n: int = 5):
    rag: CombinedRAG = app.state.rag
    return {"top_congested": rag.gov_top_congested(top_n)}

@app.get("/trips/sample")
async def sample_trips(limit: int = 10):
    df: pd.DataFrame = app.state.user_data
    sample = df.head(limit).to_dict(orient='records')
    return {"sample_trips": sample}

# if __name__ == "__main__":
#     uvicorn.run("mchat:app", host="127.0.0.1", port=8001, reload=True)
