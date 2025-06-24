from flask import Flask, render_template, request, jsonify
import math
import logging
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# AI Configuration - Using Gemini only
AI_CONFIG = {
    'google_ai_key': os.environ.get('GOOGLE_AI_API_KEY', 'your-google-ai-api-key-here'),
    'model': 'gemini-2.0-flash-exp'  # Updated to use the latest Gemini model
}

class GeminiAIExplainer:
    """Google Gemini AI-powered explanation generator for wireless network calculations"""
    
    def __init__(self):
        self.api_key = AI_CONFIG.get('google_ai_key')
        self.models = [
            'gemini-1.5-flash',      # Most reliable
            'gemini-1.5-pro',        # High quality
            'gemini-2.0-flash-exp',  # Latest experimental
            'gemini-pro'             # Fallback
        ]
        self.working_model = None
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
    def is_available(self):
        """Check if Gemini API is available"""
        return self.api_key and self.api_key != 'your-google-ai-api-key-here'
    
    def _get_working_model(self):
        """Find the first working model"""
        if self.working_model:
            return self.working_model
            
        for model in self.models:
            if self._test_model(model):
                self.working_model = model
                app.logger.info(f"Using Gemini model: {model}")
                return model
                
        app.logger.error("No working Gemini models found")
        return None
    
    def _test_model(self, model):
        """Test if a specific model is working"""
        try:
            url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
            
            data = {
                'contents': [{
                    'parts': [{
                        'text': 'Test: Say "OK" if you can respond.'
                    }]
                }],
                'generationConfig': {
                    'maxOutputTokens': 10
                }
            }
            
            response = requests.post(url, json=data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                return 'candidates' in result and len(result['candidates']) > 0
                
        except Exception:
            pass
            
        return False
    
    def generate_explanation(self, scenario, parameters, results):
        """Generate AI-powered explanation using Google Gemini"""
        if not self.is_available():
            return self._local_explanation(scenario, parameters, results)
        
        model = self._get_working_model()
        if not model:
            return self._local_explanation(scenario, parameters, results)
            
        try:
            prompt = self._create_explanation_prompt(scenario, parameters, results)
            
            url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
            
            data = {
                'contents': [{
                    'parts': [{
                        'text': prompt
                    }]
                }],
                'generationConfig': {
                    'temperature': 0.7,
                    'topK': 40,
                    'topP': 0.95,
                    'maxOutputTokens': 2048,  # Increased from 1000
                    'responseMimeType': 'text/plain'
                },
                'safetySettings': [
                    {
                        'category': 'HARM_CATEGORY_HARASSMENT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    },
                    {
                        'category': 'HARM_CATEGORY_HATE_SPEECH',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    },
                    {
                        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    },
                    {
                        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    }
                ]
            }
            
            response = requests.post(url, json=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content
                else:
                    app.logger.warning("No content in Gemini response")
                    return self._local_explanation(scenario, parameters, results)
            else:
                app.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return self._local_explanation(scenario, parameters, results)
                
        except Exception as e:
            app.logger.error(f"Gemini explanation error: {str(e)}")
            return self._local_explanation(scenario, parameters, results)
    
    def answer_question(self, scenario, parameters, results, question, conversation_history=None):
        """Answer specific questions about the calculation results"""
        if not self.is_available():
            return "AI service not available. Please configure GOOGLE_AI_API_KEY."
        
        model = self._get_working_model()
        if not model:
            return "AI service temporarily unavailable. Please try again later."
            
        try:
            prompt = self._create_question_prompt(scenario, parameters, results, question, conversation_history)
            
            url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
            
            data = {
                'contents': [{
                    'parts': [{
                        'text': prompt
                    }]
                }],
                'generationConfig': {
                    'temperature': 0.8,
                    'topK': 40,
                    'topP': 0.95,
                    'maxOutputTokens': 1500,  # Increased from 800
                    'responseMimeType': 'text/plain'
                },
                'safetySettings': [
                    {
                        'category': 'HARM_CATEGORY_HARASSMENT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    },
                    {
                        'category': 'HARM_CATEGORY_HATE_SPEECH', 
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    },
                    {
                        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    },
                    {
                        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                    }
                ]
            }
            
            response = requests.post(url, json=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content
                else:
                    return "I couldn't generate a response. Please try rephrasing your question."
            else:
                app.logger.error(f"Gemini Q&A error: {response.status_code} - {response.text}")
                return "Sorry, I'm having trouble processing your question right now."
                
        except Exception as e:
            app.logger.error(f"Gemini Q&A error: {str(e)}")
            return "An error occurred while processing your question."
    
    def _create_explanation_prompt(self, scenario, parameters, results):
        """Create a detailed prompt for explaining calculation results"""
        return f"""
You are an expert wireless communications engineer. Explain the following wireless network calculation results in a clear, educational manner.

SCENARIO: {scenario.replace('_', ' ').title()}

INPUT PARAMETERS:
{json.dumps(parameters, indent=2)}

CALCULATED RESULTS:
{json.dumps(results, indent=2)}

Please provide a comprehensive explanation that includes:

1. **Overview**: Brief summary of what this calculation accomplishes
2. **Parameter Analysis**: Explain what each key input parameter means and its typical values
3. **Results Breakdown**: Interpret the calculated results and their significance
4. **Engineering Insights**: Discuss trade-offs, optimizations, and practical considerations
5. **Real-World Context**: How these results apply to actual wireless systems

Format your response with clear sections using markdown-style headers and bullet points.
Keep explanations accessible to engineering students while maintaining technical accuracy.
Include relevant formulas when helpful, and mention typical industry values for context.
"""

    def _create_question_prompt(self, scenario, parameters, results, question, conversation_history):
        """Create prompt for answering specific questions about results"""
        history_text = ""
        if conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_history[-3:]  # Last 3 exchanges
            ])
        
        return f"""
You are an expert wireless communications engineer. Answer the following question about these wireless network calculation results.

SCENARIO: {scenario.replace('_', ' ').title()}

INPUT PARAMETERS:
{json.dumps(parameters, indent=2)}

CALCULATED RESULTS:  
{json.dumps(results, indent=2)}{history_text}

CURRENT QUESTION: {question}

Please provide a clear, accurate answer that:
- Directly addresses the specific question asked
- References the relevant parameters and results
- Explains any technical concepts mentioned
- Provides practical insights when appropriate
- Suggests follow-up considerations if relevant

Keep your answer concise but comprehensive, suitable for an engineering student.
"""

    def _local_explanation(self, scenario, parameters, results):
        """Fallback local explanation when AI service is unavailable"""
        explanations = {
            'wireless_communication': f"""
            **Wireless Communication System Analysis**
            
            Your system processes data through multiple stages:
            • **Sampling**: {parameters.get('sampling_frequency', 0)/1000:.1f} kHz rate
            • **Quantization**: {parameters.get('quantization_levels', 0)} levels = {results.get('bits_per_sample', 0):.1f} bits/sample
            • **Final Rate**: {results.get('burst_format_rate', 0)/1000:.1f} kbps output
            
            Each stage affects throughput and quality. Higher quantization and coding rates improve quality but reduce throughput.
            """,
            
            'ofdm_systems': f"""
            **OFDM System Performance**
            
            Your {parameters.get('modulation_order', 0)}-QAM system achieves:
            • **Spectral Efficiency**: {results.get('spectral_efficiency', 0):.2f} bits/s/Hz
            • **Maximum Capacity**: {results.get('max_capacity', 0)/1e6:.1f} Mbps
            • **Resource Blocks**: {parameters.get('num_parallel_rb', 0)} parallel blocks
            
            Higher modulation increases capacity but requires better signal quality.
            """,
            
            'link_budget': f"""
            **Link Budget Analysis**
            
            Your {parameters.get('frequency_mhz', 0)} MHz link:
            • **Path Loss**: {results.get('fspl_db', 0):.1f} dB
            • **Received Power**: {results.get('rss_dbm', 0):.1f} dBm
            • **Link Quality**: {results.get('link_quality', 'Unknown')}
            
            Path loss increases 6dB for every doubling of distance or frequency.
            """,
            
            'cellular_design': f"""
            **Cellular Network Design**
            
            Your network design:
            • **Coverage**: {results.get('total_num_cells', 0)} cells for {parameters.get('total_area', 0)} km²
            • **Capacity**: {results.get('max_num_users', 0):,} total users
            • **Cluster Size**: N = {results.get('cluster_size_n', 0)}
            
            Smaller cells increase capacity but require more infrastructure.
            """
        }
        return explanations.get(scenario, "Calculation completed successfully.")

# Initialize Gemini AI explainer
ai_explainer = GeminiAIExplainer()

class WirelessNetworkCalculator:
    """Main calculator class for all wireless network scenarios"""
    
    @staticmethod
    def wireless_communication_system(params):
        """
        Scenario 1: Wireless Communication System
        Computes rates at output of each block in the communication chain
        """
        try:
            # Input parameters
            source_rate = float(params.get('source_rate', 0))  # bits/second
            sampling_frequency = float(params.get('sampling_frequency', 0))  # Hz
            quantization_levels = int(params.get('quantization_levels', 2))
            source_coding_efficiency = float(params.get('source_coding_efficiency', 1.0))
            channel_coding_rate = float(params.get('channel_coding_rate', 1.0))
            interleaver_block_size = int(params.get('interleaver_block_size', 1))
            burst_format_overhead = float(params.get('burst_format_overhead', 0))
            
            # Calculations for each block
            results = {}
            
            # 1. Sampler output rate
            sampler_rate = sampling_frequency
            results['sampler_rate'] = sampler_rate
            
            # 2. Quantizer output rate
            bits_per_sample = math.log2(quantization_levels)
            quantizer_rate = sampler_rate * bits_per_sample
            results['quantizer_rate'] = quantizer_rate
            results['bits_per_sample'] = bits_per_sample
            
            # 3. Source encoder output rate
            source_encoder_rate = quantizer_rate * source_coding_efficiency
            results['source_encoder_rate'] = source_encoder_rate
            
            # 4. Channel encoder output rate
            channel_encoder_rate = source_encoder_rate / channel_coding_rate
            results['channel_encoder_rate'] = channel_encoder_rate
            
            # 5. Interleaver output rate (same as input for block interleavers)
            interleaver_rate = channel_encoder_rate
            results['interleaver_rate'] = interleaver_rate
            
            # 6. Burst formatting output rate
            burst_format_rate = interleaver_rate * (1 + burst_format_overhead)
            results['burst_format_rate'] = burst_format_rate
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                The wireless communication system processes data through multiple stages:
                
                1. **Sampler**: Converts analog signal to discrete samples at {sampler_rate} Hz
                2. **Quantizer**: Each sample is quantized to {quantization_levels} levels ({bits_per_sample:.1f} bits/sample)
                   Output rate: {quantizer_rate:.2f} bits/second
                3. **Source Encoder**: Reduces redundancy with efficiency {source_coding_efficiency}
                   Output rate: {source_encoder_rate:.2f} bits/second
                4. **Channel Encoder**: Adds redundancy for error correction (rate {channel_coding_rate})
                   Output rate: {channel_encoder_rate:.2f} bits/second
                5. **Interleaver**: Rearranges data to combat burst errors
                   Output rate: {interleaver_rate:.2f} bits/second (unchanged)
                6. **Burst Formatter**: Adds framing overhead ({burst_format_overhead*100:.1f}%)
                   Final output rate: {burst_format_rate:.2f} bits/second
                """
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def ofdm_systems(params):
        """
        Scenario 2: OFDM Systems
        Calculates data rates for various OFDM components
        """
        try:
            # Input parameters
            subcarrier_spacing = float(params.get('subcarrier_spacing', 15000))  # Hz
            symbol_duration = float(params.get('symbol_duration', 0))  # seconds
            cyclic_prefix_ratio = float(params.get('cyclic_prefix_ratio', 0.25))
            modulation_order = int(params.get('modulation_order', 4))  # QPSK=4, 16QAM=16, etc.
            coding_rate = float(params.get('coding_rate', 0.5))
            num_subcarriers_per_rb = int(params.get('num_subcarriers_per_rb', 12))
            num_symbols_per_rb = int(params.get('num_symbols_per_rb', 14))
            num_parallel_rb = int(params.get('num_parallel_rb', 1))
            bandwidth = float(params.get('bandwidth', 20e6))  # Hz
            
            results = {}
            
            # Calculate symbol duration if not provided
            if symbol_duration == 0:
                symbol_duration = 1 / subcarrier_spacing
            
            # Total symbol duration including cyclic prefix
            total_symbol_duration = symbol_duration * (1 + cyclic_prefix_ratio)
            results['total_symbol_duration'] = total_symbol_duration
            
            # Bits per symbol for modulation
            bits_per_symbol = math.log2(modulation_order)
            results['bits_per_symbol'] = bits_per_symbol
            
            # 1. Resource Element (RE) data rate
            re_data_rate = bits_per_symbol * coding_rate / total_symbol_duration
            results['re_data_rate'] = re_data_rate
            
            # 2. OFDM Symbol data rate
            ofdm_symbol_rate = re_data_rate * num_subcarriers_per_rb
            results['ofdm_symbol_rate'] = ofdm_symbol_rate
            
            # 3. Resource Block (RB) data rate
            rb_data_rate = re_data_rate * num_subcarriers_per_rb * num_symbols_per_rb
            results['rb_data_rate'] = rb_data_rate
            
            # 4. Maximum transmission capacity using parallel RBs
            max_capacity = rb_data_rate * num_parallel_rb
            results['max_capacity'] = max_capacity
            
            # 5. Spectral efficiency
            spectral_efficiency = max_capacity / bandwidth
            results['spectral_efficiency'] = spectral_efficiency
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                OFDM System Analysis:
                
                **Basic Parameters:**
                - Subcarrier spacing: {subcarrier_spacing/1000:.1f} kHz
                - Symbol duration: {symbol_duration*1e6:.1f} μs
                - Total symbol duration (with CP): {total_symbol_duration*1e6:.1f} μs
                - Modulation: {modulation_order}-QAM ({bits_per_symbol} bits/symbol)
                - Coding rate: {coding_rate}
                
                **Data Rates:**
                1. **Resource Element**: {re_data_rate/1000:.2f} kbps per RE
                2. **OFDM Symbol**: {ofdm_symbol_rate/1e6:.2f} Mbps ({num_subcarriers_per_rb} subcarriers)
                3. **Resource Block**: {rb_data_rate/1e6:.2f} Mbps ({num_subcarriers_per_rb}×{num_symbols_per_rb} REs)
                4. **Maximum Capacity**: {max_capacity/1e6:.2f} Mbps ({num_parallel_rb} parallel RBs)
                5. **Spectral Efficiency**: {spectral_efficiency:.2f} bits/s/Hz
                
                The spectral efficiency indicates how efficiently the system uses the available bandwidth.
                """
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def link_budget_calculation(params):
        """
        Scenario 3: Link Budget Calculation
        Computes transmitted power and received signal strength in flat environment
        """
        try:
            # Input parameters
            transmit_power_dbm = float(params.get('transmit_power_dbm', 30))
            transmit_antenna_gain_db = float(params.get('transmit_antenna_gain_db', 15))
            receive_antenna_gain_db = float(params.get('receive_antenna_gain_db', 10))
            frequency_mhz = float(params.get('frequency_mhz', 2400))
            distance_km = float(params.get('distance_km', 1))
            cable_loss_db = float(params.get('cable_loss_db', 2))
            misc_losses_db = float(params.get('misc_losses_db', 3))
            atmospheric_loss_db = float(params.get('atmospheric_loss_db', 0))
            
            results = {}
            
            # Convert units for calculations
            frequency_hz = frequency_mhz * 1e6
            distance_m = distance_km * 1000
            
            # Free space path loss calculation
            # FSPL (dB) = 32.45 + 20*log10(f_MHz) + 20*log10(d_km)
            fspl_db = 32.45 + 20 * math.log10(frequency_mhz) + 20 * math.log10(distance_km)
            results['fspl_db'] = fspl_db
            
            # Effective Isotropic Radiated Power (EIRP)
            eirp_dbm = transmit_power_dbm + transmit_antenna_gain_db - cable_loss_db
            results['eirp_dbm'] = eirp_dbm
            
            # Total path losses
            total_path_loss_db = fspl_db + atmospheric_loss_db + misc_losses_db
            results['total_path_loss_db'] = total_path_loss_db
            
            # Received Signal Strength (RSS)
            rss_dbm = eirp_dbm - total_path_loss_db + receive_antenna_gain_db
            results['rss_dbm'] = rss_dbm
            
            # Convert to watts for additional insight
            transmit_power_w = 10 ** ((transmit_power_dbm - 30) / 10)
            rss_w = 10 ** ((rss_dbm - 30) / 10)
            results['transmit_power_w'] = transmit_power_w
            results['rss_w'] = rss_w
            
            # Calculate link margins (using typical values)
            typical_sensitivity_dbm = -95  # Typical receiver sensitivity
            typical_fade_margin_db = 10    # Typical fade margin
            
            sensitivity_margin_db = rss_dbm - typical_sensitivity_dbm
            fade_link_margin_db = sensitivity_margin_db - typical_fade_margin_db
            
            results['sensitivity_margin_db'] = sensitivity_margin_db
            results['fade_link_margin_db'] = fade_link_margin_db
            
            # Link quality assessment
            if fade_link_margin_db >= 15:
                link_quality = "Excellent"
                quality_color = "success"
            elif fade_link_margin_db >= 10:
                link_quality = "Good"
                quality_color = "info"
            elif fade_link_margin_db >= 5:
                link_quality = "Adequate"
                quality_color = "warning"
            elif fade_link_margin_db >= 0:
                link_quality = "Marginal"
                quality_color = "warning"
            else:
                link_quality = "Insufficient"
                quality_color = "danger"
            
            results['link_quality'] = link_quality
            results['quality_color'] = quality_color
            
            # Maximum theoretical range (when RSS = sensitivity)
            if rss_dbm > typical_sensitivity_dbm:
                range_improvement_db = rss_dbm - typical_sensitivity_dbm
                max_range_multiplier = 10 ** (range_improvement_db / 20)
                max_range_km = distance_km * max_range_multiplier
                results['max_theoretical_range_km'] = max_range_km
            else:
                results['max_theoretical_range_km'] = distance_km
            
            # Path loss breakdown
            results['cable_loss_db'] = cable_loss_db
            results['atmospheric_loss_db'] = atmospheric_loss_db
            results['total_gains_db'] = transmit_antenna_gain_db + receive_antenna_gain_db
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                **Link Budget Analysis for {frequency_mhz} MHz at {distance_km} km:**
                
                **Transmitter Configuration:**
                - Transmit Power: {transmit_power_dbm} dBm ({transmit_power_w*1000:.2f} mW)
                - Antenna Gain: {transmit_antenna_gain_db} dBi
                - Cable Loss: {cable_loss_db} dB
                - **EIRP: {eirp_dbm:.2f} dBm**
                
                **Propagation Analysis:**
                - Free Space Path Loss: {fspl_db:.2f} dB
                - Atmospheric Loss: {atmospheric_loss_db} dB
                - Miscellaneous Losses: {misc_losses_db} dB
                - **Total Path Loss: {total_path_loss_db:.2f} dB**
                
                **Receiver Performance:**
                - Antenna Gain: {receive_antenna_gain_db} dBi
                - **Received Signal Strength: {rss_dbm:.2f} dBm**
                - Power Level: {rss_w*1e12:.2f} pW
                
                **Link Quality Assessment:**
                - Sensitivity Margin: {sensitivity_margin_db:.2f} dB
                - Link Margin (with fade): {fade_link_margin_db:.2f} dB
                - **Overall Quality: {link_quality}**
                - Max Range: {results['max_theoretical_range_km']:.2f} km
                
                **Key Insights:**
                - Path loss increases 6 dB for every doubling of distance or frequency
                - FSPL dominates the link budget at {frequency_mhz} MHz
                - {"✅ Excellent link with high reliability" if fade_link_margin_db > 15 else "✅ Good link performance" if fade_link_margin_db > 10 else "⚠️ Adequate but consider improvements" if fade_link_margin_db > 5 else "⚠️ Marginal - increase power or reduce distance" if fade_link_margin_db > 0 else "❌ Insufficient link - major improvements required"}
                - Total antenna gains: {results['total_gains_db']} dB
                """
            }
        except Exception as e:
            app.logger.error(f"Link budget calculation error: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    
    @staticmethod
    def cellular_system_design(params):
        """
        Scenario 4: Cellular System Design - UPDATED VERSION
        Designs a cellular network following proper cellular engineering principles
        """
        try:
            # Input parameters
            time_slots_per_carrier = int(params.get('time_slots_per_carrier', 8))  # GSM standard
            total_area = float(params.get('total_area', 100))  # km²
            max_num_users = int(params.get('max_num_users', 10000))
            avg_call_duration = float(params.get('avg_call_duration', 120))  # seconds
            avg_call_rate_per_user = float(params.get('avg_call_rate_per_user', 2))  # calls per day
            gos = float(params.get('gos', 0.02))  # Grade of Service (2%)
            sir_db = float(params.get('sir_db', 18))  # Signal-to-Interference Ratio in dB
            p0_dbm = float(params.get('p0_dbm', 30))  # Transmit power in dBm
            receiver_sensitivity_dbm = float(params.get('receiver_sensitivity_dbm', -95))  # dBm
            d0_m = float(params.get('d0_m', 1000))  # Reference distance in meters
            path_loss_exponent = float(params.get('path_loss_exponent', 4))  # Typical urban environment
            
            results = {}
            
            # Constants
            cluster_sizes = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]
            nb = 6  # Number of base stations in first tier
            
            # Unit conversions
            def db_to_watt(db_value):
                return 10 ** (db_value / 10)
            
            def dbm_to_watt(dbm_value):
                return 10 ** ((dbm_value - 30) / 10)
            
            # Convert dBm to watts
            p0_watt = dbm_to_watt(p0_dbm)
            receiver_sensitivity_watt = dbm_to_watt(receiver_sensitivity_dbm)
            sir_ratio = db_to_watt(sir_db)
            
            results['p0_watt'] = p0_watt
            results['receiver_sensitivity_watt'] = receiver_sensitivity_watt
            results['sir_ratio'] = sir_ratio
            
            # 1. Calculate maximum distance for reliable communication
            max_distance = ((receiver_sensitivity_watt / p0_watt) ** (-1 / path_loss_exponent)) * d0_m
            results['max_distance_m'] = max_distance
            results['max_distance_km'] = max_distance / 1000
            
            # 2. Calculate maximum cell size (hexagonal area)
            # Area = 3 * sqrt(3) / 2 * R^2
            max_cell_size_m2 = (3 * math.sqrt(3) / 2) * (max_distance ** 2)
            max_cell_size_km2 = max_cell_size_m2 / 1e6
            results['max_cell_size_km2'] = max_cell_size_km2
            
            # 3. Calculate total number of cells
            total_num_cells = math.ceil(total_area / max_cell_size_km2)
            results['total_num_cells'] = total_num_cells
            
            # 4. Calculate traffic per user (in Erlangs)
            # Traffic = (calls/day) * (duration/call) / (seconds/day)
            traffic_per_user = (avg_call_rate_per_user * avg_call_duration) / (24 * 60 * 60)
            results['traffic_per_user_erlang'] = traffic_per_user
            
            # 5. Calculate total system traffic
            traffic_all_system = traffic_per_user * max_num_users
            results['traffic_all_system_erlang'] = traffic_all_system
            
            # 6. Calculate traffic load for each cell
            traffic_load_each_cell = traffic_all_system / total_num_cells
            results['traffic_load_each_cell_erlang'] = traffic_load_each_cell
            
            # 7. Calculate cluster size N
            # N >= (SIR * NB)^(2/n) / 3
            x = (sir_ratio * nb) ** (2 / path_loss_exponent) / 3
            cluster_size_n = next((n for n in cluster_sizes if n >= x), cluster_sizes[-1])
            results['cluster_size_n'] = cluster_size_n
            results['x_value'] = x
            
            # 8. Erlang B table lookup for number of channels
            # Simplified Erlang B table (GOS% : [traffic values for channels 1,2,3...])
            erlang_b_table = {
                "0.1%": [0.001, 0.046, 0.194, 0.439, 0.762, 1.100, 1.600, 2.100, 2.600, 3.100, 3.700,
            4.200, 4.800, 5.400, 6.100, 6.700, 7.400, 8.000, 8.700, 9.400, 10.100,
            10.800, 11.500, 12.200, 13.000, 13.700, 14.400, 15.200, 15.900, 16.700,
            17.400, 18.200, 19.000, 19.700, 20.500, 21.300, 22.100, 22.900, 23.700,
            24.400, 25.200, 26.000, 26.800, 27.600, 28.400],
    0.2: [0.002, 0.065, 0.249, 0.535, 0.900, 1.300, 1.800, 
        2.300, 2.900, 3.400, 4.000, 4.600, 5.300, 5.900, 6.600, 
        7.300, 7.900, 8.600, 9.400, 10.100, 10.800, 11.500, 12.300,
            13.000, 13.800, 14.500, 15.300, 16.100, 16.800, 17.600, 18.400,
            19.200, 20.000, 20.800, 21.600, 22.400, 23.200, 24.000, 24.800, 25.600,
            26.400, 27.200, 28.100, 28.900, 29.7],
    0.5: [0.005, 0.105, 0.349, 0.701, 1.132, 1.600, 2.200, 2.700,
            3.300, 4.000, 4.600, 5.300, 6.000, 6.700, 7.400, 8.100, 8.800, 
            9.600, 10.300, 11.100, 11.900, 12.600, 13.400, 14.200, 15.000, 15.800,
            16.600, 17.400, 18.200, 19.000, 19.900, 20.700, 21.500, 22.300, 23.200,
            24.000, 24.800, 25.700, 26.500, 27.400, 28.200, 29.100, 29.900, 30.800, 31.7],
    1: [0.010, 0.153, 0.455, 0.869, 1.361, 1.900, 2.500, 3.100, 3.800, 4.500, 5.200,
        5.900, 6.600, 7.400, 8.100, 8.900, 9.700, 10.400, 11.200, 12.000, 12.800, 13.700,
            14.500, 15.300, 16.100, 17.000, 17.800, 18.600, 19.500, 20.300, 21.200, 22.000, 22.900,
            23.800, 24.600, 25.500, 26.400, 27.300, 28.100, 29.000, 29.900, 30.800, 31.700, 32.500, 33.4],
    1.2: [0.012, 0.168, 0.489, 0.922, 1.431, 2.000, 2.600, 3.200, 3.900, 
        4.600, 5.300, 6.100, 6.800, 7.600, 8.300, 9.100, 9.900, 10.700, 11.500,
            12.300, 13.100, 14.000, 14.800, 15.600, 16.500, 17.300, 18.200, 19.000,
            19.900, 20.700, 21.600, 22.500, 23.300, 24.200, 25.100, 26.000, 26.800,
            27.700, 28.600, 29.500, 30.400, 31.300, 32.200, 33.100, 34.000],
    1.3: [0.013, 0.176, 0.505, 0.946, 1.464, 2.000, 2.700, 3.300, 4.000,
            4.700, 5.400, 6.100, 6.900, 7.700, 8.400, 9.200, 10.000, 10.800, 
            11.600, 12.400, 13.300, 14.100, 14.900, 15.800, 16.600, 17.500, 
            18.300, 19.200, 20.000, 20.900, 21.800, 22.600, 23.500, 24.400, 
            25.300, 26.200, 27.000, 27.900, 28.800, 29.700, 30.600, 31.500,
            32.400, 33.300, 34.200],
    1.5: [0.020, 0.190, 0.530, 0.990, 1.520, 2.100, 2.700, 3.400, 4.100, 
        4.800, 5.500, 6.300, 7.000, 7.800, 8.600, 9.400, 10.200, 11.000, 11.800,
            12.600, 13.500, 14.300, 15.200, 16.000, 16.900, 17.700, 18.600, 19.500,
            20.300, 21.200, 22.100, 22.900, 23.800, 24.700, 25.600, 26.500, 27.400,
            28.300, 29.200, 30.100, 31.000, 31.900, 32.800, 33.700, 34.600],
    2: [0.020, 0.223, 0.602, 1.092, 1.657, 2.300, 2.900, 3.600, 4.300, 5.100,
            5.800, 6.600, 7.400, 8.200, 9.000, 9.800, 10.700, 11.500, 12.300, 13.200, 
            14.000, 14.900, 15.800, 16.600, 17.500, 18.400, 19.300, 20.200, 21.000, 21.900,
            22.800, 23.700, 24.600, 25.500, 26.400, 27.300, 28.300, 29.200, 30.100, 31.000, 
            31.900, 32.800, 33.800, 34.700, 35.600],
    3: [0.031, 0.282, 0.715, 1.259, 1.875, 2.500, 3.200, 4.000, 4.700, 5.500,
            6.300, 7.100, 8.000, 8.800, 9.600, 10.500, 11.400, 12.200, 13.100, 14.000,
            14.900, 15.800, 16.700, 17.600, 18.500, 19.400, 20.300, 21.200, 22.100, 23.100,
            24.000, 24.900, 25.800, 26.800, 27.700, 28.600, 29.600, 30.500, 31.500, 32.400,
            33.400, 34.300, 35.300, 36.200, 37.200],
    5: [0.053, 0.381, 0.899, 1.525, 2.218, 3.000, 3.700, 4.500, 5.400, 6.200,
            7.100, 8.000, 8.800, 9.700, 10.600, 11.500, 12.500, 13.400, 14.300, 15.200,
            16.200, 17.100, 18.100, 19.000, 20.000, 20.900, 21.900, 22.900, 23.800,
            24.800, 25.800, 26.700, 27.700, 28.700, 29.700, 30.700, 31.600, 32.600,
            33.600, 34.600, 35.600, 36.600, 37.600, 38.600, 39.600],
    7: [0.075, 0.470, 1.057, 1.748, 2.504, 3.300, 4.100, 5.000, 5.900, 6.800,
            7.700, 8.600, 9.500, 10.500, 11.400, 12.400, 13.400, 14.300, 15.300, 16.300,
            17.300, 18.200, 19.200, 20.200, 21.200, 22.200, 23.200, 24.200, 25.200, 26.200,
            27.200, 28.200, 29.300, 30.300, 31.300, 32.300, 33.300, 34.400, 35.400, 36.400,
            37.400, 38.400, 39.500, 40.500, 41.500],
    10: [0.111, 0.595, 1.271, 2.045, 2.881, 3.800, 4.700, 5.600, 6.500, 7.500, 
        8.500, 9.500, 10.500, 11.500, 12.500, 13.500, 14.500, 15.500, 16.600, 17.600,
            18.700, 19.700, 20.700, 21.800, 22.800, 23.900, 24.900, 26.000, 27.100, 28.100, 
            29.200, 30.200, 31.300, 32.400, 33.400, 34.500, 35.600, 36.600, 37.700, 38.800,
            39.900, 40.900, 42.000, 43.100, 44.200],
    15: [0.176, 0.796, 1.602, 2.501, 3.454, 4.400, 5.500, 6.500, 7.600, 8.600, 9.700,
            10.800, 11.900, 13.000, 14.100, 15.200, 16.300, 17.400, 18.500, 19.600, 20.800,
            21.900, 23.000, 24.200, 25.300, 26.400, 27.600, 28.700, 29.900, 31.000, 32.100,
            33.300, 34.400, 35.600, 36.700, 37.900, 39.000, 40.200, 41.300, 42.500, 43.600,
            44.800, 45.900, 47.100, 48.200],
    20: [0.250, 1.000, 1.930, 2.950, 4.010, 5.100, 6.200, 7.400, 8.500, 9.700, 10.900, 
        12.000, 13.200, 14.400, 15.600, 16.800, 18.000, 19.200, 20.400, 21.600, 22.800, 24.100,
            25.300, 26.500, 27.700, 28.900, 30.200, 31.400, 32.600, 33.800, 35.100, 36.300, 37.500, 
            38.800, 40.000, 41.200, 42.400, 43.700, 44.900, 46.100, 47.400, 48.600, 49.900, 51.100, 52.300],
    30: [0.429, 1.450, 2.633, 3.890, 5.189, 6.500, 7.900, 9.200, 10.600, 12.000, 13.300,
            14.700, 16.100, 17.500, 18.900, 20.300, 21.700, 23.100, 24.500, 25.900, 27.300, 28.700,
            30.100, 31.600, 33.000, 34.400, 35.800, 37.200, 38.600, 40.000, 41.500, 42.900, 44.300,
            45.700, 47.100, 48.600, 50.000, 51.400, 52.800, 54.200, 55.700, 57.100, 58.500, 59.900, 61.300]
};

            
            
            
            
            
            
            # Find GOS percentage key (convert to nearest available)
            gos_percent = gos * 100
            if gos_percent <= 1.5:
                gos_key = 1.0
            elif gos_percent <= 3.5:
                gos_key = 2.0
            else:
                gos_key = 5.0
            
            results['gos_key_used'] = gos_key
            
            # Find number of channels required using Erlang B table
            traffic_values = erlang_b_table[gos_key]
            num_channels_required = 1
            for i, traffic_capacity in enumerate(traffic_values):
                if traffic_capacity >= traffic_load_each_cell:
                    num_channels_required = i + 1
                    break
            else:
                num_channels_required = len(traffic_values)
            
            results['num_channels_required'] = num_channels_required
            results['traffic_capacity_found'] = traffic_values[num_channels_required - 1] if num_channels_required <= len(traffic_values) else traffic_values[-1]
            
            # 9. Calculate number of carriers per cell
            num_carriers_per_cell = math.ceil(num_channels_required / time_slots_per_carrier)
            results['num_carriers_per_cell'] = num_carriers_per_cell
            
            # 10. Calculate total carriers in system
            num_carriers_in_system = num_carriers_per_cell * cluster_size_n
            results['num_carriers_in_system'] = num_carriers_in_system
            
            # Additional system parameters
            results['total_channels_in_system'] = num_channels_required * total_num_cells
            results['spectrum_efficiency'] = max_num_users / total_area  # users per km²
            results['cell_density'] = total_num_cells / total_area  # cells per km²
            
            # System capacity analysis
            actual_system_capacity = num_channels_required * total_num_cells
            channel_utilization = (traffic_load_each_cell * total_num_cells) / actual_system_capacity
            results['actual_system_capacity_channels'] = actual_system_capacity
            results['channel_utilization'] = channel_utilization
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                **Advanced Cellular System Design Analysis:**
                
                **1. Coverage Analysis:**
                - Maximum reliable distance: {max_distance/1000:.3f} km
                - Maximum cell size: {max_cell_size_km2:.3f} km²
                - Total cells required: {total_num_cells} cells
                - Cell density: {results['cell_density']:.2f} cells/km²
                
                **2. Link Budget Analysis:**
                - Transmit power: {p0_dbm} dBm ({p0_watt*1000:.2f} mW)
                - Receiver sensitivity: {receiver_sensitivity_dbm} dBm ({receiver_sensitivity_watt*1e12:.2f} pW)
                - Path loss exponent: {path_loss_exponent}
                - Reference distance: {d0_m/1000:.1f} km
                
                **3. Interference Analysis:**
                - Required SIR: {sir_db} dB ({sir_ratio:.1f} ratio)
                - First-tier interferers: {nb}
                - Calculated cluster size: N = {cluster_size_n}
                - Minimum required: {x:.2f}
                
                **4. Traffic Engineering:**
                - Traffic per user: {traffic_per_user*1000:.2f} mErlang
                - Total system traffic: {traffic_all_system:.2f} Erlang
                - Traffic per cell: {traffic_load_each_cell:.3f} Erlang
                - Grade of Service: {gos*100:.1f}% (using {gos_key}% table)
                
                **5. Channel Assignment:**
                - Channels per cell: {num_channels_required}
                - Time slots per carrier: {time_slots_per_carrier}
                - Carriers per cell: {num_carriers_per_cell}
                - Total system carriers: {num_carriers_in_system}
                - Channel utilization: {channel_utilization*100:.1f}%
                
                **6. System Performance:**
                - Total system users: {max_num_users:,}
                - Actual channel capacity: {actual_system_capacity}
                - Spectrum efficiency: {results['spectrum_efficiency']:.1f} users/km²
                - Traffic capacity found: {results['traffic_capacity_found']:.3f} Erlang
                
                **Key Engineering Insights:**
                - Cell size is limited by link budget (max distance = {max_distance/1000:.3f} km)
                - Cluster size N={cluster_size_n} provides adequate SIR of {sir_db} dB
                - System can handle {traffic_all_system:.2f} Erlang with {gos*100:.1f}% blocking
                - Each cell requires {num_carriers_per_cell} carrier(s) for {num_channels_required} channels
                - Total frequency reuse efficiency: 1/{cluster_size_n} = {1/cluster_size_n:.3f}
                """
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Initialize calculator
calculator = WirelessNetworkCalculator()

@app.route('/')
def index():
    """Main page with scenario selection"""
    return render_template('index.html')

@app.route('/scenario/<int:scenario_id>')
def scenario(scenario_id):
    """Individual scenario pages"""
    scenarios = {
        1: 'wireless_communication',
        2: 'ofdm_systems', 
        3: 'link_budget',
        4: 'cellular_design'
    }
    
    if scenario_id in scenarios:
        return render_template(f'{scenarios[scenario_id]}.html')
    else:
        return "Scenario not found", 404

@app.route('/calculate/<scenario>', methods=['POST'])
def calculate(scenario):
    """API endpoint for calculations with Gemini AI-powered explanations"""
    try:
        data = request.get_json()
        
        if scenario == 'wireless_communication':
            result = calculator.wireless_communication_system(data)
        elif scenario == 'ofdm_systems':
            result = calculator.ofdm_systems(data)
        elif scenario == 'link_budget':
            result = calculator.link_budget_calculation(data)
        elif scenario == 'cellular_design':
            result = calculator.cellular_system_design(data)
        else:
            return jsonify({'success': False, 'error': 'Invalid scenario'})
        
        # Enhance explanation with Gemini AI
        if result['success']:
            try:
                ai_explanation = ai_explainer.generate_explanation(scenario, data, result['results'])
                if ai_explanation and len(ai_explanation.strip()) > 50:  # Valid AI response
                    result['explanation'] = ai_explanation
                    result['ai_powered'] = True
                    result['ai_service'] = 'gemini'
            except Exception as e:
                app.logger.warning(f"Gemini explanation failed, using local: {str(e)}")
                # Keep the original explanation from calculator
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Calculation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask questions about calculation results using Gemini AI"""
    try:
        data = request.get_json()
        scenario = data.get('scenario')
        parameters = data.get('parameters', {})
        results = data.get('results', {})
        question = data.get('question', '')
        conversation_history = data.get('conversation_history', [])
        
        if not question.strip():
            return jsonify({'success': False, 'error': 'Question cannot be empty'})
        
        # Get answer from Gemini AI
        answer = ai_explainer.answer_question(scenario, parameters, results, question, conversation_history)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'question': question,
            'ai_service': 'gemini',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Question answering error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/explain', methods=['POST'])
def explain_results():
    """Enhanced Gemini AI-powered explanation endpoint"""
    try:
        data = request.get_json()
        scenario = data.get('scenario')
        results = data.get('results', {})
        parameters = data.get('parameters', {})
        
        # Generate enhanced explanation using Gemini AI
        explanation = ai_explainer.generate_explanation(scenario, parameters, results)
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'ai_powered': ai_explainer.is_available(),
            'ai_service': 'gemini',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current Gemini AI configuration status"""
    return jsonify({
        'ai_service': 'gemini',
        'ai_available': ai_explainer.is_available(),
        'model': AI_CONFIG.get('model'),
        'status': 'ready' if ai_explainer.is_available() else 'api_key_required'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)