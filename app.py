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
            • **Coverage**: {results.get('num_cells', 0)} cells for {parameters.get('coverage_area_km2', 0)} km²
            • **Capacity**: {results.get('total_system_users', 0):,} total users
            • **Spectrum**: {results.get('spectrum_per_cell_mhz', 0):.1f} MHz per cell
            
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
        Scenario 4: Cellular System Design
        Designs a cellular network based on user parameters
        """
        try:
            # Input parameters
            coverage_area_km2 = float(params.get('coverage_area_km2', 100))  # km²
            cell_radius_km = float(params.get('cell_radius_km', 1))  # km
            frequency_reuse_factor = int(params.get('frequency_reuse_factor', 7))
            total_spectrum_mhz = float(params.get('total_spectrum_mhz', 25))  # MHz
            users_per_cell = int(params.get('users_per_cell', 1000))
            traffic_per_user_erlang = float(params.get('traffic_per_user_erlang', 0.025))  # Erlang
            blocking_probability = float(params.get('blocking_probability', 0.02))  # 2%
            sectorization = int(params.get('sectorization', 1))  # sectors per cell
            
            results = {}
            
            # Cell area calculation (hexagonal cells)
            cell_area_km2 = 2.6 * (cell_radius_km ** 2)  # Hexagonal area = 3√3/2 * R²
            results['cell_area_km2'] = cell_area_km2
            
            # Number of cells required
            num_cells = math.ceil(coverage_area_km2 / cell_area_km2)
            results['num_cells'] = num_cells
            
            # Spectrum per cell
            spectrum_per_cell_mhz = total_spectrum_mhz / frequency_reuse_factor
            results['spectrum_per_cell_mhz'] = spectrum_per_cell_mhz
            
            # Traffic analysis
            total_traffic_per_cell = users_per_cell * traffic_per_user_erlang
            results['total_traffic_per_cell'] = total_traffic_per_cell
            
            # Erlang B calculation (approximation for required channels)
            # Using approximation: C ≈ A + C₀√A where C₀ depends on blocking probability
            c0_factor = 1.8 if blocking_probability <= 0.02 else 1.5
            channels_per_cell = math.ceil(total_traffic_per_cell + c0_factor * math.sqrt(total_traffic_per_cell))
            results['channels_per_cell'] = channels_per_cell
            
            # With sectorization
            if sectorization > 1:
                traffic_per_sector = total_traffic_per_cell / sectorization
                channels_per_sector = math.ceil(traffic_per_sector + c0_factor * math.sqrt(traffic_per_sector))
                total_channels_per_cell = channels_per_sector * sectorization
                results['sectorization'] = sectorization
                results['traffic_per_sector'] = traffic_per_sector
                results['channels_per_sector'] = channels_per_sector
                results['total_channels_per_cell_sectored'] = total_channels_per_cell
            
            # System capacity
            total_system_users = num_cells * users_per_cell
            total_system_channels = num_cells * channels_per_cell
            results['total_system_users'] = total_system_users
            results['total_system_channels'] = total_system_channels
            
            # Spectral efficiency
            channels_per_mhz = channels_per_cell / spectrum_per_cell_mhz if spectrum_per_cell_mhz > 0 else 0
            results['channels_per_mhz'] = channels_per_mhz
            
            # Cell site density
            cell_density_per_km2 = num_cells / coverage_area_km2
            results['cell_density_per_km2'] = cell_density_per_km2
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                Cellular System Design Analysis:
                
                **Coverage Design:**
                - Total coverage area: {coverage_area_km2} km²
                - Cell radius: {cell_radius_km} km
                - Cell area: {cell_area_km2:.2f} km² (hexagonal)
                - Number of cells required: {num_cells} cells
                - Cell density: {cell_density_per_km2:.2f} cells/km²
                
                **Frequency Planning:**
                - Total spectrum: {total_spectrum_mhz} MHz
                - Frequency reuse factor: {frequency_reuse_factor}
                - Spectrum per cell: {spectrum_per_cell_mhz:.2f} MHz
                
                **Traffic Engineering:**
                - Users per cell: {users_per_cell}
                - Traffic per user: {traffic_per_user_erlang} Erlang
                - Total traffic per cell: {total_traffic_per_cell:.2f} Erlang
                - Required channels per cell: {channels_per_cell}
                - Blocking probability: {blocking_probability*100:.1f}%
                
                {"**Sectorization:**" if sectorization > 1 else ""}
                {f"- Sectors per cell: {sectorization}" if sectorization > 1 else ""}
                {f"- Traffic per sector: {results.get('traffic_per_sector', 0):.2f} Erlang" if sectorization > 1 else ""}
                {f"- Channels per sector: {results.get('channels_per_sector', 0)}" if sectorization > 1 else ""}
                
                **System Capacity:**
                - Total system users: {total_system_users:,}
                - Total system channels: {total_system_channels}
                - Spectral efficiency: {channels_per_mhz:.2f} channels/MHz
                
                This design provides {blocking_probability*100:.1f}% blocking probability with {frequency_reuse_factor}-cell reuse pattern.
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