# AI-Powered Wireless Network Design Tool

## 🚀 Project Overview

This is a comprehensive web-based application for wireless and mobile network calculations, developed for the **Wireless and Mobile Networks course (ENCS5323)** at Birzeit University. The application provides advanced calculations with AI-powered explanations for various wireless communication scenarios.

## 📡 Features

### Core Calculation Modules

1. **Wireless Communication System**
   - Sampler rate calculations
   - Quantizer data rate analysis
   - Source encoder efficiency
   - Channel encoder rate computation
   - Interleaver processing
   - Burst formatter overhead analysis

2. **OFDM Systems Analysis**
   - Resource element (RE) data rate calculations
   - Resource block (RB) performance analysis
   - Symbol duration and cyclic prefix optimization
   - Spectral efficiency computation
   - Maximum capacity analysis

3. **Link Budget Calculation**
   - Free Space Path Loss (FSPL) computation
   - Effective Isotropic Radiated Power (EIRP) analysis
   - Received Signal Strength (RSS) calculation
   - Link margin and quality assessment
   - Support for multiple transmission scenarios (WiFi, Cellular, Satellite, Microwave)

4. **Cellular System Design**
   - Coverage area planning
   - Frequency reuse factor optimization
   - Traffic engineering (Erlang B model)
   - Cell capacity and user distribution
   - Sectorization analysis

### 🤖 AI-Powered Features

- **Google Gemini Integration**: Advanced AI explanations for all calculations
- **Interactive Q&A**: Ask specific questions about calculation results
- **Educational Insights**: Technical explanations suitable for engineering students
- **Real-world Context**: Practical applications and industry standards

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **AI Integration**: Google Gemini API
- **Styling**: Custom CSS with gradient backgrounds and modern UI
- **Icons**: Font Awesome
- **Environment Management**: python-dotenv

## 📋 Prerequisites

- Python 3.7 or higher
- Google AI API key (for Gemini integration)
- Modern web browser with JavaScript enabled

## 🔧 Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd wireless-network-design
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_AI_API_KEY=your_google_ai_api_key_here
   ```

   To get a Google AI API key:
   - Visit [Google AI Studio](https://ai.google.dev/)
   - Create a new API key
   - Copy the key to your `.env` file

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   
   Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
wireless-network-design/
├── app.py                          # Main Flask application
├── wsgi.py                         # WSGI configuration for deployment
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .env                           # Environment variables (create this)
└── templates/                     # HTML templates
    ├── base.html                  # Base template with common styling
    ├── index.html                 # Home page
    ├── wireless_communication.html # Wireless communication calculator
    ├── ofdm_systems.html          # OFDM systems calculator
    ├── link_budget.html           # Link budget calculator
    └── cellular_design.html       # Cellular design calculator
```

## 🧮 Calculation Formulas

### Wireless Communication System
- **Quantizer Rate**: `Rate = fs × log₂(L)` where L = quantization levels
- **Channel Encoder**: `Rate = Rsource / Rcode`
- **Burst Formatter**: `Rate = Rinterleaver × (1 + overhead)`

### OFDM Systems
- **Symbol Duration**: `Ts = 1 / Δf`
- **Total Symbol Time**: `Ttotal = Ts × (1 + CP)`
- **RE Data Rate**: `RRE = (bits/symbol × Rcode) / Ttotal`
- **Spectral Efficiency**: `η = Rtotal / Btotal`

### Link Budget
- **Free Space Path Loss**: `FSPL = 32.45 + 20log₁₀(f) + 20log₁₀(d)`
- **EIRP**: `EIRP = Ptx + Gtx - Lcable`
- **Received Signal Strength**: `RSS = EIRP - FSPL + Grx - Lmisc`

### Cellular Design
- **Cell Area**: `Acell = 2.6 × R²` (hexagonal cells)
- **Number of Cells**: `Ncells = Atotal / Acell`
- **Traffic Engineering**: `C ≈ A + C₀√A` (Erlang B approximation)

## 🎯 Usage Examples

### Link Budget Calculation
1. Select transmission type (WiFi, Cellular, Satellite, Microwave)
2. Input transmitter parameters (power, antenna gain, frequency)
3. Set distance and environmental factors
4. Get detailed analysis with AI explanations

### OFDM Analysis
1. Configure subcarrier spacing and symbol duration
2. Set modulation order and coding rate
3. Define resource block structure
4. Analyze data rates and spectral efficiency

## 🔍 AI Features Usage

1. **Automatic Explanations**: After each calculation, AI provides detailed explanations
2. **Interactive Q&A**: Use the chat interface to ask specific questions
3. **Educational Context**: Get explanations suitable for engineering students
4. **Real-world Applications**: Learn how calculations apply to actual systems

## 🎓 Educational Value

This tool is designed for:
- **Engineering Students**: Learn wireless communication principles
- **Network Engineers**: Quick calculations and analysis
- **Researchers**: Prototype and analyze wireless systems
- **Educators**: Teaching aid with visual results and explanations

## 🔧 Configuration

### AI Configuration
- The application automatically tries multiple Gemini models for best performance
- Fallback explanations are provided when AI is unavailable
- API usage is optimized to minimize costs

### Customization
- Modify calculation parameters in `app.py`
- Update UI styling in template files
- Add new calculation scenarios by extending the `WirelessNetworkCalculator` class

## 📄 License

This project is developed for educational purposes as part of the ENCS5323 course at Birzeit University.

