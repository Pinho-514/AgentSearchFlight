import os, json, requests
from datetime import datetime
from crewai import Agent, Task, Crew, entrypoint
from crewai.tools import BaseTool
import os


# --- helpers ---------------------------------------------------------------
companies = {"LA": "LATAM", "G3": "GOL", "AD": "Azul", "2Z": "Voepass"}

def _to_iso(date_br: str) -> str:
    """Convert dd/MM/yyyy → yyyy-MM-dd (Amadeus expects ISO)."""
    return datetime.strptime(date_br, "%d/%m/%Y").strftime("%Y-%m-%d")

def _get_rate(cur: str) -> float:
    """Convert any currency to BRL (two-decimal precision)."""
    url = "https://api.exchangeratesapi.io/v1/latest"
    r = requests.get(
        url,
        params={
            "access_key": os.getenv("EXR_API_KEY"),
            "base": cur,
            "symbols": "BRL",
        },
        timeout=10,
    )
    return r.json()["rates"]["BRL"]

def _amadeus_token() -> str:
    r = requests.post(
        "https://test.api.amadeus.com/v1/security/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("AMADEUS_CLIENT_ID"),
            "client_secret": os.getenv("AMADEUS_CLIENT_CRED"),
        },
        timeout=10,
    )
    return r.json()["access_token"]

def _search_leg(token: str, origin: str, dest: str, date: str) -> list[dict]:
    """Return raw Amadeus offers (max 20). """
    r = requests.get(
        "https://test.api.amadeus.com/v2/shopping/flight-offers",
        params={
            "originLocationCode": origin,
            "destinationLocationCode": dest,
            "departureDate": date,
            "adults": 1,
            "max": 20,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    return r.json()["data"]


# --- CrewAI tool -----------------------------------------------------------
class FlightSearchTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Tool expected parameters: origin (airpoirt code), destination (airport code), departure_date, return_date" 
        "Return TWO arrays (outbound, return) with up to 20 options each, "
        "sorted by total BRL price ascending. Each element has: "
        "company, departure_time, arrival_time, stops, seats_remaining, price_brl."
    )

    def _run(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str
    ) -> str:                       # returns JSON text
        token = _amadeus_token()
        print(f'Tool started with parameters: origin ({origin}) | destination ({destination}) | departure_date ({departure_date}) | return_date ({return_date}) ')
        legs = {
            "outbound": _search_leg(token, origin, destination, _to_iso(departure_date)),
            "return":   _search_leg(token, destination, origin, _to_iso(return_date)),
        }

        rates, result = {}, {"outbound": [], "return": []}
        for leg_name, offers in legs.items():
            for o in offers:
                cur = o["price"]["currency"]
                if cur not in rates:
                    rates[cur] = _get_rate(cur)
                seg = o["itineraries"][0]["segments"][0]
                result[leg_name].append(
                    {
                        "company": companies.get(seg["carrierCode"], seg["carrierCode"]),
                        "departure_time": seg["departure"]["at"],
                        "arrival_time": seg["arrival"]["at"],
                        "stops": seg["numberOfStops"],
                        "seats_remaining": o["numberOfBookableSeats"],
                        "price_brl": round(float(o["price"]["grandTotal"]) * rates[cur], 2),
                    }
                )
            result[leg_name].sort(key=lambda v: v["price_brl"])

        return json.dumps(result, ensure_ascii=False)


flight_tool = FlightSearchTool()

# --- agents ----------------------------------------------------------------
extractor = Agent(
    role="Email parser",
    goal="Identify origin, destination, departure_date and return_date from a Portuguese e-mail",
    backstory="Regex and date-recognition expert for Brazilian travellers"
)

planner = Agent(
    role="Flight planner",
    goal=(
        "Call search_flights, analyse its 20-option arrays and reply with ONLY "
        "a JSON object containing keys 'outbound_top3' and 'return_top3'."
        "Outbound flights should favour morning departures (07-11 h); "
        "return flights should favour afternoon departures (17-21 h). "
        "Inside each leg, rank by a composite of lower price, fewer stops "
        "and preferred departure time as described."
    ),
    backstory="Flight-data analyst specialised in price optimisation",
    tools=[flight_tool]
)

# --- tasks -----------------------------------------------------------------
email_task = Task(
    description=(
        "From {{email_text}} extract JSON: "
        "{origin:'IATA', destination:'IATA', departure_date:'dd/mm/yyyy', "
        "return_date:'dd/mm/yyyy'}."
    ),
    agent=extractor,
    expected_output="JSON with keys origin, destination, departure_date, return_date",
)

plan_task = Task(
    description=(
        "Receive the JSON, call search_flights, keep only the three cheapest "
        "for each leg, and output a JSON with keys outbound_top3 and return_top3."
    ),
    agent=planner,
    expected_output="JSON with keys outbound_top3 and return_top3, each containing 3 flight objects",
)

crew = Crew(tasks=[email_task, plan_task],verbose=True)



@entrypoint
def run(email_text: str):
    """
    Main entry called by the CrewAI platform.
    Expects one parameter: the raw e-mail text.
    """
    return crew.kickoff(inputs={"email_text": email_text})