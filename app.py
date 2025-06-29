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
            'gemini-1.5-flash',      
            'gemini-1.5-pro',        
            'gemini-2.0-flash-exp',  
            'gemini-pro'            
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
            headers = {
                'Content-Type': 'application/json',
                'key': f'Bearer {self.api_key}'
            }
            
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
                    'maxOutputTokens': 2048,  
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
            
            response = requests.post(url, json=data,headers=headers , timeout=15)
            
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
        Uses Nyquist rate sampling (fs = 2 * bandwidth)
        """
        try:
            # Input parameters
            signal_bandwidth = float(params.get('signal_bandwidth', 0))  # Hz
            sampling_frequency = float(params.get('sampling_frequency', 0))  # Hz (Nyquist rate)
            quantization_levels = int(params.get('quantization_levels', 2))
            source_coding_efficiency = float(params.get('source_coding_efficiency', 1.0))
            channel_coding_rate = float(params.get('channel_coding_rate', 1.0))
            interleaver_block_size = int(params.get('interleaver_block_size', 1))
            burst_format_overhead = float(params.get('burst_format_overhead', 0))
            
            # Verify Nyquist rate
            nyquist_rate = 2 * signal_bandwidth
            if abs(sampling_frequency - nyquist_rate) > 0.1:  # Allow small floating point differences
                sampling_frequency = nyquist_rate
            
            # Calculations for each block
            results = {}
            
            # Store input parameters for reference
            results['signal_bandwidth'] = signal_bandwidth
            results['nyquist_rate'] = nyquist_rate
            
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
                The wireless communication system processes data through multiple stages using Nyquist rate sampling:
                
                **Input Signal**: Bandwidth = {signal_bandwidth:.0f} Hz
                
                1. **Sampler**: Converts analog signal to discrete samples at Nyquist rate
                   Nyquist Rate = 2 × Bandwidth = {nyquist_rate:.0f} Hz
                   Sampling Rate: {sampler_rate:.0f} Hz
                   
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
                   
                **Note**: The Nyquist-Shannon sampling theorem ensures perfect reconstruction of the original signal
                when sampling at or above the Nyquist rate (2 × maximum frequency component).
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
        Scenario 3: Link Budget Calculation using provided formulas
        Calculates transmitted and received power using the specific formulas:
        1. Tₙ (dB) = 10 × log₁₀(T)
        2. R (dB) = 10 × log₁₀(Data Rate in bps)
        3. Pᵣ = M - 228.6 + Tₙ + N + R + Eb/No
        4. Pₜ = Pᵣ + Lp + Lf + Lo + Lfm - Gt - Gr - Ar
        5. Watts = 10^(dB / 10)
        """
        try:
            # Extract parameters
            lp_db = float(params.get('lp_db', 10.0))  # Path Loss
            frequency_mhz = float(params.get('frequency_mhz', 1000.0))  # Frequency
            gt_db = float(params.get('gt_db', 32.0))  # Transmitter Antenna Gain
            gr_db = float(params.get('gr_db', 32.0))  # Receiver Antenna Gain
            data_rate_kbps = float(params.get('data_rate_kbps', 1231.0))  # Data Rate
            lf_db = float(params.get('lf_db', 1.0))  # Antenna Feed Line Loss
            lo_db = float(params.get('lo_db', 1.0))  # Other Losses
            lfm_db = float(params.get('lfm_db', 1.0))  # Fade Margin
            ar_db = float(params.get('ar_db', 2.0))  # Receiver Amplifier Gain
            n_db = float(params.get('n_db', 3.0))  # Noise Figure
            noise_temp_k = float(params.get('noise_temp_k', 3.0))  # Noise Temperature
            m_db = float(params.get('m_db', 1.0))  # Link Margin
            ebno_db = float(params.get('ebno_db', 1.0))  # Eb/No
            
            # Step 1: Calculate Noise Temperature in dB
            # Tₙ (dB) = 10 × log₁₀(T)
            tn_db = 10 * math.log10(noise_temp_k)
            
            # Step 2: Calculate Data Rate in dB
            # R (dB) = 10 × log₁₀(Data Rate in bps)
            data_rate_bps = data_rate_kbps * 1000  # Convert kbps to bps
            r_db = 10 * math.log10(data_rate_bps)
            
            # Step 3: Calculate Received Power
            # Pᵣ = M - 228.6 + Tₙ + N + R + Eb/No
            pr_db = m_db - 228.6 + tn_db + n_db + r_db + ebno_db
            
            # Step 4: Calculate Transmitted Power
            # Pₜ = Pᵣ + Lp + Lf + Lo + Lfm - Gt - Gr - Ar
            pt_db = pr_db + lp_db + lf_db + lo_db + lfm_db - gt_db - gr_db - ar_db
            
            # Step 5: Convert to Watts
            # Watts = 10^(dB / 10)
            pr_w = 10 ** (pr_db / 10)
            pt_w = 10 ** (pt_db / 10)
            
            # Prepare results
            results = {
                'pr_db': pr_db,
                'pt_db': pt_db,
                'pr_w': pr_w,
                'pt_w': pt_w,
                'tn_db': tn_db,
                'r_db': r_db,
                'data_rate_bps': data_rate_bps,
                'noise_temp_k': noise_temp_k,
                'lp_db': lp_db,
                'lf_db': lf_db,
                'lo_db': lo_db,
                'lfm_db': lfm_db,
                'gt_db': gt_db,
                'gr_db': gr_db,
                'ar_db': ar_db,
                'n_db': n_db,
                'm_db': m_db,
                'ebno_db': ebno_db
            }
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                **Link Budget Calculation Results:**
                
                **Input Parameters:**
                - Path Loss (Lp): {lp_db} dB
                - Frequency: {frequency_mhz} MHz
                - Transmitter Antenna Gain (Gt): {gt_db} dB
                - Receiver Antenna Gain (Gr): {gr_db} dB
                - Data Rate: {data_rate_kbps} kbps
                - Feed Line Loss (Lf): {lf_db} dB
                - Other Losses (Lo): {lo_db} dB
                - Fade Margin (Lfm): {lfm_db} dB
                - Receiver Amplifier Gain (Ar): {ar_db} dB
                - Noise Figure (N): {n_db} dB
                - Noise Temperature: {noise_temp_k} K
                - Link Margin (M): {m_db} dB
                - Eb/No: {ebno_db} dB
                
                **Calculation Steps:**
                1. Tₙ = 10 × log₁₀({noise_temp_k}) = {tn_db:.2f} dB
                2. R = 10 × log₁₀({data_rate_bps:.0f}) = {r_db:.2f} dB
                3. Pᵣ = {m_db} - 228.6 + {tn_db:.2f} + {n_db} + {r_db:.2f} + {ebno_db} = **{pr_db:.2f} dB**
                4. Pₜ = {pr_db:.2f} + {lp_db} + {lf_db} + {lo_db} + {lfm_db} - {gt_db} - {gr_db} - {ar_db} = **{pt_db:.2f} dB**
                5. Power conversion: 10^(dB/10)
                
                **Final Results:**
                - **Power at Receiver (Pr): {pr_db:.2f} dB = {pr_w:.2e} W**
                - **Power at Transmitter (Pt): {pt_db:.2f} dB = {pt_w:.2e} W**
                """
            }
            
        except Exception as e:
            app.logger.error(f"Link budget calculation error: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    
    @staticmethod
    def cellular_system_design(params):
        """
        Scenario 4: Cellular System Design - CORRECTED EQUATION COMPLIANT VERSION
        Follows the EXACT equations with realistic parameter handling
        """
        try:
            # Input parameters with corrected defaults
            time_slots_per_carrier = int(params.get('time_slots_per_carrier', 8))
            total_area = float(params.get('total_area', 100))  # km²
            max_num_users = int(params.get('max_num_users', 10000))
            avg_call_duration = float(params.get('avg_call_duration', 180))  # seconds - CORRECTED
            avg_call_rate_per_user = float(params.get('avg_call_rate_per_user', 3))  # calls per day - CORRECTED
            gos = float(params.get('gos', 0.02))  # Grade of Service (2%)
            sir_db = float(params.get('sir_db', 18))  # Signal-to-Interference Ratio in dB
            p0_dbm = float(params.get('p0_dbm', 43))  # Transmit power in dBm - CORRECTED
            receiver_sensitivity_dbm = float(params.get('receiver_sensitivity_dbm', -104))  # dBm - CORRECTED
            d0_m = float(params.get('d0_m', 1000))  # Reference distance in meters
            path_loss_exponent = float(params.get('path_loss_exponent', 4))
            frequency_mhz = float(params.get('frequency_mhz', 900))  # Operating frequency
            
            results = {}
            
            # Input validation
            if p0_dbm <= receiver_sensitivity_dbm:
                return {'success': False, 'error': f'TX Power ({p0_dbm} dBm) must be greater than RX Sensitivity ({receiver_sensitivity_dbm} dBm)'}
            
            link_budget_db = p0_dbm - receiver_sensitivity_dbm
            if link_budget_db < 50:
                app.logger.warning(f'Very low link budget: {link_budget_db:.1f} dB')
            elif link_budget_db > 200:
                app.logger.warning(f'Very high link budget: {link_budget_db:.1f} dB')
            
            # Constants
            cluster_sizes = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]
            nb = 6  # Number of base stations in first tier (as specified)
            
            # Convert dBm to Watts for power calculations
            def dbm_to_watt(dbm_value):
                return 10 ** ((dbm_value - 30) / 10)
            
            p0_watt = dbm_to_watt(p0_dbm)
            receiver_sensitivity_watt = dbm_to_watt(receiver_sensitivity_dbm)
            
            results['p0_watt'] = p0_watt
            results['receiver_sensitivity_watt'] = receiver_sensitivity_watt
            results['link_budget_db'] = link_budget_db
            
            # 1. EXACT Link Budget Equation: dmax = ((Psens/P0)^(-1/n)) × d0
            power_ratio = receiver_sensitivity_watt / p0_watt
            dmax = ((power_ratio) ** (-1 / path_loss_exponent)) * d0_m
            
            # Add practical limits to prevent unrealistic results
            dmax = min(dmax, 50000)  # Max 50km for any cellular system
            dmax = max(dmax, 100)    # Min 100m to prevent zero-area cells
            
            results['power_ratio'] = power_ratio
            results['dmax_m'] = dmax
            results['dmax_km'] = dmax / 1000
            
            # 2. EXACT Cell Area Equation: Cell Area = (3√3/2) × dmax²
            cell_area_m2 = (3 * math.sqrt(3) / 2) * (dmax ** 2)
            cell_area_km2 = cell_area_m2 / 1e6
            
            results['cell_area_m2'] = cell_area_m2
            results['cell_area_km2'] = cell_area_km2
            
            # 3. Calculate total number of cells needed for coverage
            total_num_cells = max(1, math.ceil(total_area / cell_area_km2))
            results['total_num_cells'] = total_num_cells
            
            # 4. EXACT Traffic Engineering Equations
            # Auser = (calls/day × duration) / 86400
            traffic_per_user = (avg_call_rate_per_user * avg_call_duration) / 86400
            results['traffic_per_user_erlang'] = traffic_per_user
            
            # Asystem = Auser × Nusers
            traffic_all_system = traffic_per_user * max_num_users
            results['traffic_all_system_erlang'] = traffic_all_system
            
            # Acell = Asystem / Ncells
            traffic_load_each_cell = traffic_all_system / total_num_cells
            results['traffic_load_each_cell_erlang'] = traffic_load_each_cell
            
            # 5. EXACT Interference Analysis: N ≥ (SIR × NB)^(2/n) / 3
            sir_linear = 10 ** (sir_db / 10)  # Convert dB to linear
            cluster_size_calculated = ((sir_linear * nb) ** (2 / path_loss_exponent)) / 3
            
            # Find the next valid cluster size
            cluster_size_n = next((n for n in cluster_sizes if n >= cluster_size_calculated), cluster_sizes[-1])
            
            results['sir_linear'] = sir_linear
            results['nb'] = nb
            results['cluster_size_calculated'] = cluster_size_calculated
            results['cluster_size_n'] = cluster_size_n
            
            # 6. Channel Assignment using Erlang B table
            erlang_b_table = {
                0.5: [0.005, 0.105, 0.349, 0.701, 1.132, 1.627, 2.157, 2.718, 3.305, 3.917, 4.550, 5.204, 5.876, 6.564, 7.269, 7.989, 8.722, 9.468, 10.226, 10.994, 11.774, 12.564, 13.365, 14.175, 14.993],
                1.0: [0.010, 0.153, 0.455, 0.869, 1.361, 1.909, 2.501, 3.128, 3.783, 4.462, 5.160, 5.876, 6.608, 7.354, 8.114, 8.886, 9.669, 10.463, 11.267, 12.081, 12.904, 13.736, 14.576, 15.425, 16.282],
                2.0: [0.020, 0.223, 0.602, 1.092, 1.657, 2.276, 2.935, 3.627, 4.345, 5.086, 5.847, 6.624, 7.416, 8.221, 9.037, 9.864, 10.700, 11.546, 12.401, 13.263, 14.133, 15.011, 15.896, 16.789, 17.689],
                3.0: [0.031, 0.282, 0.715, 1.259, 1.875, 2.543, 3.250, 3.987, 4.748, 5.529, 6.328, 7.141, 7.968, 8.806, 9.654, 10.512, 11.378, 12.252, 13.134, 14.023, 14.920, 15.823, 16.734, 17.651, 18.575],
                5.0: [0.053, 0.381, 0.899, 1.525, 2.218, 2.960, 3.738, 4.543, 5.370, 6.216, 7.078, 7.953, 8.840, 9.737, 10.643, 11.558, 12.481, 13.411, 14.348, 15.292, 16.244, 17.202, 18.166, 19.136, 20.112]
            }
            
            # Find appropriate GOS key
            gos_percent = gos * 100
            if gos_percent <= 0.75:
                gos_key = 0.5
            elif gos_percent <= 1.5:
                gos_key = 1.0
            elif gos_percent <= 2.5:
                gos_key = 2.0
            elif gos_percent <= 4.0:
                gos_key = 3.0
            else:
                gos_key = 5.0
            
            results['gos_key_used'] = gos_key
            
            # Find number of channels using Erlang B table
            traffic_values = erlang_b_table[gos_key]
            num_channels_required = 1
            
            for i, traffic_capacity in enumerate(traffic_values):
                if traffic_capacity >= traffic_load_each_cell:
                    num_channels_required = i + 1
                    break
            else:
                # If traffic exceeds table, use approximation
                num_channels_required = min(100, len(traffic_values) + int(traffic_load_each_cell * 2))
            
            results['num_channels_required'] = num_channels_required
            results['traffic_capacity_used'] = traffic_values[min(num_channels_required - 1, len(traffic_values) - 1)]
            
            # 7. EXACT Channel Assignment Equations
            # Carriers = ⌈Channels / Time_Slots⌉
            num_carriers_per_cell = math.ceil(num_channels_required / time_slots_per_carrier)
            results['num_carriers_per_cell'] = num_carriers_per_cell
            
            # System Carriers = Carriers × N (where N is cluster size, NOT total cells)
            num_carriers_in_system = num_carriers_per_cell * cluster_size_n
            results['num_carriers_in_system'] = num_carriers_in_system
            
            # Additional calculations for completeness
            total_channels_in_system = num_channels_required * total_num_cells
            results['total_channels_in_system'] = total_channels_in_system
            
            # Performance metrics
            actual_traffic_capacity = results['traffic_capacity_used'] * total_num_cells
            channel_utilization = min(1.0, traffic_all_system / actual_traffic_capacity) if actual_traffic_capacity > 0 else 0
            users_per_cell = max_num_users / total_num_cells
            spectrum_efficiency = max_num_users / total_area
            cell_density = total_num_cells / total_area
            frequency_reuse_efficiency = 1 / cluster_size_n
            
            results['actual_traffic_capacity_erlang'] = actual_traffic_capacity
            results['channel_utilization'] = channel_utilization
            results['users_per_cell'] = users_per_cell
            results['spectrum_efficiency'] = spectrum_efficiency
            results['cell_density'] = cell_density
            results['frequency_reuse_efficiency'] = frequency_reuse_efficiency
            
            # Store intermediate calculations for verification
            results['calculations_verification'] = {
                'link_budget_formula': f"dmax = ((Psens/P0)^(-1/n)) × d0 = (({receiver_sensitivity_watt:.2e}/{p0_watt:.2e})^(-1/{path_loss_exponent})) × {d0_m} = {dmax:.2f} m",
                'cell_area_formula': f"Area = (3√3/2) × dmax² = (3×√3/2) × {dmax:.2f}² = {cell_area_km2:.3f} km²",
                'traffic_user_formula': f"Auser = (calls/day × duration) / 86400 = ({avg_call_rate_per_user} × {avg_call_duration}) / 86400 = {traffic_per_user:.6f} Erlang",
                'traffic_system_formula': f"Asystem = Auser × Nusers = {traffic_per_user:.6f} × {max_num_users} = {traffic_all_system:.3f} Erlang",
                'traffic_cell_formula': f"Acell = Asystem / Ncells = {traffic_all_system:.3f} / {total_num_cells} = {traffic_load_each_cell:.6f} Erlang",
                'interference_formula': f"N ≥ (SIR × NB)^(2/n) / 3 = ({sir_linear:.1f} × {nb})^(2/{path_loss_exponent}) / 3 = {cluster_size_calculated:.3f}",
                'carriers_formula': f"Carriers = ⌈{num_channels_required} / {time_slots_per_carrier}⌉ = {num_carriers_per_cell}",
                'system_carriers_formula': f"System Carriers = Carriers × N = {num_carriers_per_cell} × {cluster_size_n} = {num_carriers_in_system}"
            }
            
            # Add warnings for unusual results
            warnings = []
            if dmax > 20000:
                warnings.append(f"Very large cell radius ({dmax/1000:.1f} km) - check power settings")
            if dmax < 500:
                warnings.append(f"Very small cell radius ({dmax/1000:.2f} km) - may be unrealistic")
            if traffic_load_each_cell > 20:
                warnings.append(f"High traffic per cell ({traffic_load_each_cell:.2f} E) - consider more cells")
            if num_channels_required > 50:
                warnings.append(f"Many channels required ({num_channels_required}) - check traffic assumptions")
            
            results['warnings'] = warnings
            
            return {
                'success': True,
                'results': results,
                'explanation': f"""
                **CORRECTED Cellular System Design Results:**
                
                **Input Parameters Validation:**
                - Link Budget: {link_budget_db:.1f} dB (P0: {p0_dbm} dBm, Psens: {receiver_sensitivity_dbm} dBm)
                - Traffic per user: {traffic_per_user*1000:.2f} mErlang (corrected calculation)
                - Frequency: {frequency_mhz} MHz, Path loss exponent: {path_loss_exponent}
                
                **1. Link Budget (EXACT Formula):**
                dmax = ((Psens/P0)^(-1/n)) × d0
                Power ratio = {receiver_sensitivity_watt:.2e} / {p0_watt:.2e} = {power_ratio:.2e}
                **dmax = {dmax/1000:.3f} km** (limited to realistic range)
                
                **2. Cell Area (EXACT Formula):**
                Cell Area = (3√3/2) × dmax² = **{cell_area_km2:.3f} km²**
                
                **3. Traffic Engineering (EXACT Formulas):**
                Auser = ({avg_call_rate_per_user} × {avg_call_duration}) / 86400 = **{traffic_per_user*1000:.3f} mErlang**
                Asystem = {traffic_per_user:.6f} × {max_num_users} = **{traffic_all_system:.3f} Erlang**
                Acell = {traffic_all_system:.3f} / {total_num_cells} = **{traffic_load_each_cell:.6f} Erlang**
                
                **4. Interference Analysis (EXACT Formula):**
                N ≥ (SIR × NB)^(2/n) / 3 = ({sir_linear:.1f} × {nb})^(2/{path_loss_exponent}) / 3 = {cluster_size_calculated:.3f}
                **Selected N = {cluster_size_n}**
                
                **5. Channel Assignment (EXACT Formulas):**
                Channels per cell: **{num_channels_required}** (from Erlang B table at {gos_key}% GOS)
                Carriers per cell: **{num_carriers_per_cell}** (⌈{num_channels_required}/{time_slots_per_carrier}⌉)
                System carriers: **{num_carriers_in_system}** (Carriers × N = {num_carriers_per_cell} × {cluster_size_n})
                
                **6. System Summary:**
                - Total cells: {total_num_cells} | Users per cell: {users_per_cell:.0f}
                - System utilization: {channel_utilization*100:.1f}% | Reuse efficiency: {frequency_reuse_efficiency:.3f}
                - Cell density: {cell_density:.3f} cells/km² | Spectrum efficiency: {spectrum_efficiency:.1f} users/km²
                
                {"**⚠️ Warnings:** " + "; ".join(warnings) if warnings else "**✅ All parameters within realistic ranges**"}
                
                **Key Corrections Applied:**
                1. Matched HTML defaults to Python defaults (P0=43dBm, Psens=-104dBm)
                2. Added practical limits to prevent unrealistic cell sizes
                3. Enhanced validation and warning system
                4. Improved traffic calculations for realistic values
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