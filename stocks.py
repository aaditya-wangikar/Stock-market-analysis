# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:42:38 2025

@author: lenovo
"""
def sector_stocks(index, return_all = False):
    nifty50_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'LT.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 'ITC.NS',
        'AXISBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'ASIANPAINT.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'HCLTECH.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
        'NESTLEIND.NS', 'GRASIM.NS', 'POWERGRID.NS', 'JSWSTEEL.NS', 'TECHM.NS',
        'NTPC.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'DRREDDY.NS',
        'HEROMOTOCO.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'DIVISLAB.NS', 'TATAMOTORS.NS',
        'BPCL.NS', 'EICHERMOT.NS', 'M&M.NS', 'HINDALCO.NS', 'BHARTIARTL.NS',
        'TATASTEEL.NS', 'ONGC.NS', 'INDUSINDBK.NS', 'UPL.NS', 'SHREECEM.NS',
        'BAJAJ-AUTO.NS', 'SBILIFE.NS', 'APOLLOHOSP.NS', 'HDFCLIFE.NS', 'ICICIPRULI.NS'
    ]
    
    ai_stocks_nse = [
        "TATAELXSI.NS",      # Tata Elxsi – AI in automotive, healthcare, media
        "LTTS.NS",           # L&T Technology Services – AI + IoT + embedded
        "PERSISTENT.NS",     # Persistent Systems – AI/ML in cloud apps
        "TECHM.NS",          # Tech Mahindra – Gen AI platforms, NLP
        "INFY.NS",           # Infosys – AI for automation & business analytics
        "TCS.NS",            # TCS – AI consulting, cloud, and IP platforms
        "HCLTECH.NS",        # HCLTech – AI in cybersecurity & automation
        "COFORGE.NS",        # Coforge – AI-driven digital transformation
        "MPHASIS.NS",        # Mphasis – Applied AI for finance and insurance
        "WIPRO.NS"           # Wipro – Generative AI, ML platforms
    ]
    
    
    defence_stocks = [
    "HAL.NS",         # Hindustan Aeronautics Ltd – Fighter jets, helicopters
    "BEL.NS",         # Bharat Electronics Ltd – Radars, avionics, electronics
    "BDL.NS",         # Manufacturer of missiles: Akash, Astra, Milan, and torpedoes
    "BEML.NS",        # BEML Ltd – Military trucks, earthmovers
    "MAZDOCK.NS",     # Mazagon Dock – Warship & submarine builder
    "COCHINSHIP.NS",  # Cochin Shipyard – Naval ships, aircraft carriers
    "MTARTECH.NS",    # MTAR Technologies – Precision components for DRDO/ISRO
    "DATA PATTERNS.NS",# Data Patterns – Defense electronics and systems
    # "TANEJAERO.NS",   # Taneja Aerospace – Aircraft parts, airfield infrastructure
    "SOLARA.NS",      # Solara Active Pharma (indirectly contributes to defense health)
    "SOLARINDS.NS",   # Solar industries
    "PARAS.NS",       # Paras Defence – Space, optics, electronics
    "TITAGARH.NS",     # Titagarh Rail Systems – Defense wagons, naval components
    "DYNAMATECH.NS",  # Dynamatic Technologies – Aerospace & defense systems
    "IDEAFORGE.NS",   # ideaForge Technology – Military-grade drones
    "ASTRA.MICRO.NS", # Astra Microwave – Defense radars and RF systems
    "MIRZAINT.NS"     # Mirza International – Military boots, apparel (indirect)
    "DCXINDIA.NS"     # DCX Systems operates in system integration, cable & wire harness fabrication, and electronics for defense and aerospace sectors
    
]
    
    pharma_stocks_nse = [
        "SUNPHARMA.NS",       # Sun Pharmaceutical Industries
        "DIVISLAB.NS",        # Divi's Laboratories
        "DRREDDY.NS",         # Dr. Reddy's Laboratories
        "CIPLA.NS",           # Cipla Ltd
        "AUROPHARMA.NS",      # Aurobindo Pharma
        "LUPIN.NS",           # Lupin Ltd
        "ALKEM.NS",           # Alkem Laboratories
        "ZYDUSLIFE.NS",       # Zydus Lifesciences (formerly Cadila)
        "GLENMARK.NS",        # Glenmark Pharmaceuticals
        "BIOCON.NS",          # Biocon Ltd
        "IPCALAB.NS",         # Ipca Laboratories
        "TORNTPHARM.NS",      # Torrent Pharmaceuticals
        "ABBOTINDIA.NS",      # Abbott India
        "PFIZER.NS",          # Pfizer India
        "SANOFI.NS",          # Sanofi India
        "NATCOPHARM.NS",      # Natco Pharma
        "ERIS.NS",            # Eris Lifesciences
        "AJANTPHARM.NS",      # Ajanta Pharma
        "GLAND.NS",           # Gland Pharma
        "JUBLINGREA.NS"       # Jubilant Ingrevia (pharma + chemicals)
    ]
    
    energy_stocks_nse = [
        "RELIANCE.NS",     # Reliance Industries - Oil, gas, green energy
        "ONGC.NS",         # Oil & Natural Gas Corporation - Exploration
        "IOC.NS",          # Indian Oil Corporation - Refining & retail
        "BPCL.NS",         # Bharat Petroleum - Oil marketing
        "HPCL.NS",         # Hindustan Petroleum - Oil marketing
        "GAIL.NS",         # Gas Authority of India - Gas distribution
        "NTPC.NS",         # National Thermal Power Corp - Power generation
        "POWERGRID.NS",    # Power Grid Corp - Transmission
        "TATAPOWER.NS",    # Tata Power - Conventional + solar
        "ADANIGREEN.NS",   # Adani Green Energy - Renewable energy
        "ADANITRANS.NS",   # Adani Transmission
        "ADANIPOWER.NS",   # Adani Power - Thermal
        "NHPC.NS",         # National Hydro Power Corp - Hydro
        "SJVN.NS",         # Satluj Jal Vidyut Nigam - Hydro + solar
        "COALINDIA.NS",    # Coal India - Coal mining
        "JSWENERGY.NS",    # JSW Energy - Thermal + renewables
        "TORNTPOWER.NS",   # Torrent Power - Generation + distribution
        "KPIGREEN.NS",     # K.P.I. Green Energy - Solar developer
        "INOXWIND.NS",     # Inox Wind - Wind turbine manufacturer
        "IGL.NS",          # Indraprastha Gas Ltd - Gas distribution
        "MGL.NS"           # Mahanagar Gas Ltd - City gas distribution
    ]
    
    fmcg_stocks_nse = [
        "HINDUNILVR.NS",   # Hindustan Unilever – Market leader in personal/home care
        "ITC.NS",          # ITC – Cigarettes, foods, personal care
        "NESTLEIND.NS",    # Nestle India – Packaged foods, beverages
        "BRITANNIA.NS",    # Britannia Industries – Biscuits, dairy, bakery
        "DABUR.NS",        # Dabur India – Ayurvedic and healthcare products
        "MARICO.NS",       # Marico – Hair oil, edible oil (Parachute, Saffola)
        "COLPAL.NS",       # Colgate-Palmolive – Oral care
        "EMAMILTD.NS",     # Emami – Personal care & healthcare
        "GODREJCP.NS",     # Godrej Consumer – Household & personal care
        "BAJAJCON.NS",     # Bajaj Consumer – Hair oil
        "HATSUN.NS",       # Hatsun Agro – Dairy products
        "AVANTIFEED.NS",   # Avanti Feeds – Aqua feed (semi-FMCG)
        "HERITGFOOD.NS",   # Heritage Foods – Dairy
        "VSTIND.NS",       # VST Industries – Tobacco (smaller than ITC)
        "ZYDUSWELL.NS",    # Zydus Wellness – Nutraceuticals, wellness
        "PATANJALI.NS",    # Patanjali Foods – Edible oil, FMCG (Ruchi Soya rebranded)
        "VARUNBEVER.NS",   # Varun Beverages – PepsiCo bottling partner
        "RADICO.NS",       # Radico Khaitan – Liquor (FMCG-beverage)
        "MANPASAND.NS",    # Manpasand Beverages (low volume/illiquid)
        "TASTYBITE.NS"     # Tasty Bite – Ready-to-eat food (small-cap)
    ]
    
    banking_stocks_nse = [
        # 🔵 Private Sector Banks
        "HDFCBANK.NS",       # HDFC Bank
        "ICICIBANK.NS",      # ICICI Bank
        "AXISBANK.NS",       # Axis Bank
        "KOTAKBANK.NS",      # Kotak Mahindra Bank
        "INDUSINDBK.NS",     # IndusInd Bank
        "IDFCFIRSTB.NS",     # IDFC First Bank
        "RBLBANK.NS",        # RBL Bank
        "CSBBANK.NS",        # CSB Bank
        "DCBBANK.NS",        # DCB Bank
        "YESBANK.NS",        # Yes Bank
        "KARURVYSYA.NS",     # Karur Vysya Bank
        "SOUTHBANK.NS",      # South Indian Bank
        "CITYUNION.NS",      # City Union Bank
    
        # 🔴 Public Sector Banks (PSBs)
        "SBIN.NS",           # State Bank of India
        "BANKBARODA.NS",     # Bank of Baroda
        "PNB.NS",            # Punjab National Bank
        "UNIONBANK.NS",      # Union Bank of India
        "CANBK.NS",          # Canara Bank
        "INDIANB.NS",        # Indian Bank
        "BANKINDIA.NS",      # Bank of India
        "UCOBANK.NS",        # UCO Bank
        "IOB.NS",            # Indian Overseas Bank
        "CENTRALBK.NS",      # Central Bank of India
        "MAHABANK.NS"        # Bank of Maharashtra
    ]
    
    infra_stocks_nse = [
        # 🚧 Core Infra & EPC (Engineering, Procurement, Construction)
        "LT.NS",             # Larsen & Toubro – Engineering & construction leader
        "NBCC.NS",           # NBCC (India) – Govt infra developer
        "IRCON.NS",          # IRCON International – Railways infra
        "RVNL.NS",           # Rail Vikas Nigam Ltd – Rail infra execution
        "PNCINFRA.NS",       # PNC Infratech – Roads & highways
        "KNRCON.NS",         # KNR Constructions – Roads & irrigation
        "HGINFRA.NS",        # H.G. Infra – Highway contractor
        "NCC.NS",            # NCC Ltd – Multi-sector infrastructure
        "ASHOKA.NS",         # Ashoka Buildcon – Roads, bridges, power
        "GPTINFRA.NS",       # GPT Infraprojects – Civil & rail infrastructure
    
        # 🛣️ Roads & Highways (HAM/EPC)
        "DILIPBUILCON.NS",   # Dilip Buildcon – Large road infra player
        "IRB.NS",            # IRB Infra – Toll operator, highways
        "HGIEL.NS",          # H.G. Infra Engineering Ltd
    
        # ⚡ Power & Utility Infra
        "POWERGRID.NS",      # Power Grid Corp – Power transmission infra
        "NTPC.NS",           # NTPC – Power generation
        "RECLTD.NS",         # Rural Electrification Corp – Power finance
        "PFC.NS",            # Power Finance Corp
    
        # ⚓ Ports & Transport Infra
        "ADANIPORTS.NS",     # Adani Ports – Ports, logistics
        "CONCOR.NS",         # Container Corp – Rail logistics
        "GMRINFRA.NS",       # GMR Airports Infra – Airports, urban infra
        "APLAPOLLO.NS",      # Apollo Tubes – Structural steel (infra input)
    
        # 🏙️ Urban Infra / Smart City / Metro
        # "JMC.NS",            # JMC Projects – Urban infra, metros
        "RAJRATAN.NS",       # Rajratan Global – Tyre bead wire, industrial infra
    
        # 🧱 Materials for Infra
        "RAMCOCEM.NS",       # Ramco Cement
        "ULTRACEMCO.NS",     # UltraTech Cement
        "JKCEMENT.NS",       # JK Cement
        "STLTECH.NS",        # Sterlite Tech – Optical infra for smart cities
    ]
    
    metal_stocks_nse = [
        "TATASTEEL.NS",      # Tata Steel – Integrated steel producer
        "JSWSTEEL.NS",       # JSW Steel – Leading private sector steelmaker
        "SAIL.NS",           # Steel Authority of India – Govt-owned steel PSU
        "HINDALCO.NS",       # Hindalco – Aluminum & copper (Aditya Birla Group)
        "VEDL.NS",           # Vedanta Ltd – Diversified metals & mining
        "NMDC.NS",           # NMDC – Iron ore mining PSU
        "NATIONALUM.NS",     # National Aluminium Co. – PSU aluminum producer
        "JINDALSTEL.NS",     # Jindal Steel & Power – Steel & power
        "RATNAMANI.NS",      # Ratnamani Metals – Steel tubes & pipes
        "APLAPOLLO.NS",      # APL Apollo – Structural steel pipes
        "MOIL.NS",           # Manganese Ore India Ltd – PSU manganese miner
        "HINDZINC.NS",       # Hindustan Zinc – Zinc & silver mining (Vedanta)
        "WELCORP.NS",        # Welspun Corp – Pipes (oil & gas sector)
        "MASTEK.NS",         # (Possible misclassified – tech, not metals)
        "SHYAMMETL.NS",      # Shyam Metalics – Ferrous metals & ferroalloys
        "JSWISPL.NS",        # JSW Ispat – Steel products (subsidiary of JSW)
        "TUNGAMETAL.NS",     # Tungabhadra Steel – Small-cap alloy maker
        "MANAKALUCO.NS",     # Manaksia Aluminium – Non-ferrous
        "SANDUMA.NS",        # Sandur Manganese – Iron ore & manganese
    ]
    
    gold_etf_nse  = [
    "GOLDBEES.NS",     # Nippon India Gold BeES – Most liquid and oldest gold ETF in India
    "HDFCMFGETF.NS",   # HDFC Gold ETF – Managed by HDFC Mutual Fund, well-established
    "GOLDIETF.NS",     # ICICI Prudential Gold ETF – Popular, low tracking error
    "KOTAKGOLD.NS",    # Kotak Gold ETF – Offered by Kotak Mutual Fund
    "SBIGETS.NS",      # SBI Gold ETF – From SBI Mutual Fund, large AUM
    "GOLDSHARE.NS",    # UTI Gold ETF – Managed by UTI Mutual Fund
    "BSLGOLDETF.NS",   # Aditya Birla Sun Life Gold ETF – Competitive expense ratio
    "AXISGOLD.NS",     # Axis Gold ETF – Offered by Axis Mutual Fund
    "QGOLDHALF.NS",    # Quantum Gold ETF – Passive, low-cost, first to offer direct plan
    "IDBIGOLD.NS",     # IDBI Gold ETF – From IDBI Mutual Fund
    "IVZINGOLD.NS"     # Invesco India Gold ETF – Managed by Invesco Mutual Fund
    ]
    
    auto_stocks_nse = [
        # 🚙 Passenger & Commercial Vehicles
        "TATAMOTORS.NS",       # Tata Motors – PVs, CVs, EVs (owns Jaguar-Land Rover)
        "MAHINDRA.NS",         # Mahindra & Mahindra – SUVs, tractors, EVs
        "MARUTI.NS",           # Maruti Suzuki – India’s largest passenger car maker
        "ASHOKLEY.NS",         # Ashok Leyland – Commercial vehicles
        "EICHERMOT.NS",        # Eicher Motors – Royal Enfield & trucks (VECV)
        "FORCEMOT.NS",         # Force Motors – Commercial vehicles
        "SMLISUZU.NS",         # SML Isuzu – Trucks, buses
    
        # 🛵 Two-Wheelers & Three-Wheelers
        "BAJAJ-AUTO.NS",       # Bajaj Auto – Motorcycles & 3-wheelers
        "TVSMOTOR.NS",         # TVS Motor – 2-wheelers, electric scooters
        "HEROMOTOCO.NS",       # Hero MotoCorp – World's largest 2-wheeler maker
        "ATULAUTO.NS",         # Atul Auto – 3-wheelers
    
        # 🔋 EV & Battery Linked
        "OLECTRA.NS",          # Olectra Greentech – Electric buses
        "GREAVESCOT.NS",       # Greaves Cotton – EV powertrains
        "AMARAJABAT.NS",       # Amara Raja Batteries
        "EXIDEIND.NS",         # Exide Industries – Batteries, EV tie-ups
    
        # 🔧 Auto Ancillaries
        "BOSCHLTD.NS",         # Bosch India – Auto components, electronics
        "MOTHERSUMI.NS",       # Samvardhana Motherson – Wiring, mirrors, modules
        "SONA.BLW.NS",         # Sona BLW Precision – EV & ICE drivetrain
        "ENDURANCE.NS",        # Endurance Tech – Suspension, brakes
        "SCHAEFFLER.NS",       # Schaeffler India – Bearings, engine systems
        "VARROC.NS",           # Varroc Engineering – Lighting, EV parts
    
        # 🛞 Tyre Manufacturers
        "MRF.NS",              # MRF Ltd
        "APOLLOTYRE.NS",       # Apollo Tyres
        "CEATLTD.NS",          # CEAT
        "BALKRISIND.NS",       # Balkrishna Industries – Off-road tyres
        "JKTYRE.NS",           # JK Tyre
        "TVSSRICHAK.NS",       # TVS Srichakra
    
        # 🚜 Tractors & Agri Vehicles
        "ESCORTS.NS"           # Escorts Kubota – Tractors, railway components
    ]
    
    chemical_stocks_nse = [
        # 🔬 Specialty Chemicals
        "AARTIIND.NS",       # Aarti Industries – Specialty & pharma intermediates
        "NAVINFLUOR.NS",     # Navin Fluorine – Fluorochemicals
        "SRF.NS",            # SRF Ltd – Fluorochemicals, packaging films
        "PIIND.NS",          # PI Industries – Agrochemicals + CRAMS
        "FINEORG.NS",        # Fine Organic – Oleochemicals
        "ALKYLAMINE.NS",     # Alkyl Amines – Aliphatic amines
        "BALAMINES.NS",      # Balaji Amines – Amines & derivatives
        "DEEPAKNTR.NS",      # Deepak Nitrite – Phenol, acetone, nitrites
        "VINATIORGA.NS",     # Vinati Organics – ATBS, IBB (specialty chemicals)
        "TATACHEM.NS",       # Tata Chemicals – Soda ash, nutraceuticals
        "LAURUSLABS.NS",     # Laurus Labs – APIs & pharma chemicals
        "GUJALKALI.NS",      # Gujarat Alkalies – Caustic soda, chlorine
        "GHCL.NS",           # GHCL – Soda ash, textiles
        "CHEMPLASTS.NS",     # Chemplast Sanmar – PVC & specialty pastes
        "HEUBACHIND.NS",     # Heubach Colorants (formerly Clariant) – Pigments
        "RELCHEMQ.NS",       # Reliance Chemotex – Industrial chemicals
        "JUBLINGREA.NS",     # Jubilant Ingrevia – Chemicals, life science ingredients
    
        # 🌾 Agrochemicals
        "RALLIS.NS",         # Rallis India – Tata Group, agrochemicals
        "BHARATRAS.NS",      # Bharat Rasayan – Pesticides, technicals
        "BASF.NS",           # BASF India – Global agro and industrial chemical MNC
        "DHANUKA.NS",        # Dhanuka Agritech – Pesticides
        "INSECTICID.NS",     # Insecticides India – Agrochemicals
        "HERANBA.NS",        # Heranba Industries – Crop protection chemicals
    
        # ⚗️ Industrial / Bulk Chemicals
        "DCW.NS",            # DCW Ltd – Caustic soda, PVC
        "TANLA.NS",          # (Note: This is a tech stock, not chemical. Skip if mislisted.)
        "ALKEM.NS",          # Alkem Labs – (Mostly pharma; minor chemical exposure)
        "MEGH.NS",           # Meghmani Organics – Pigments, agrochemicals
        "KPRMILL.NS"         # (Primarily textile; check if chemical division relevant)
    ]
    
    if return_all == False:
        if index   == 1:
            return nifty50_stocks
        elif index == 2:
            return ai_stocks_nse
        elif index == 3:
            return defence_stocks
        elif index == 4:
            return pharma_stocks_nse
        elif index == 5:
            return energy_stocks_nse
        elif index == 6:
            return fmcg_stocks_nse
        elif index == 7:
            return banking_stocks_nse
        elif index == 8:
            return infra_stocks_nse
        elif index == 9:
            return metal_stocks_nse
        elif index == 10:
            return gold_etf_nse
        elif index == 11:
            return auto_stocks_nse
        elif index == 12:
            return chemical_stocks_nse
    else:
        stocks = list(set(nifty50_stocks + ai_stocks_nse + defence_stocks + pharma_stocks_nse + 
                          energy_stocks_nse + fmcg_stocks_nse + banking_stocks_nse + infra_stocks_nse + 
                         metal_stocks_nse + gold_etf_nse + auto_stocks_nse +  chemical_stocks_nse ))
        return stocks
    
    