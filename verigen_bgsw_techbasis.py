import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
 
# Load LLaMA Model & Tokenizer
model_name = "huggyllama/llama-65b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
 
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
 
# Define verification criteria with suggested default values
criteria = {
   "method": "Test, Simulation, Analysis, Review",
   "medium": "Sample prototype, SW module, Design document, Test logs",
   "environment": "Operational conditions, Test lab setup, Hardware-in-the-loop simulation",
   "acceptance_criteria": "Expected output data, Error tolerance, Pass/Fail conditions",
   "preconditions": "System power state, Initial conditions, Sensor readiness",
   "constraints": "Memory limits, Hardware availability, Execution time limits"
}
 
# Function to create a prompt
def create_prompt(requirement_text):
   prompt = f"""
   Validate the following requirement text against verification criteria.
 
   **Verification Criteria:**
   1. Method: {criteria['method']}
   2. Medium: {criteria['medium']}
   3. Environment: {criteria['environment']}
   4. Acceptance Criteria: {criteria['acceptance_criteria']}
   5. Preconditions: {criteria['preconditions']}
   6. Constraints: {criteria['constraints']}
 
   **Requirement:**
   {requirement_text}
 
   **Validation Output Format:**
   - Method: Yes/No (Reason) | Suggestion
   - Medium: Yes/No (Reason) | Suggestion
   - Environment: Yes/No (Reason) | Suggestion
   - Acceptance Criteria: Yes/No (Reason) | Suggestion
   - Preconditions: Yes/No (Reason) | Suggestion
   - Constraints: Yes/No (Reason) | Suggestion
   """
   return prompt
 
# Function to query the LLaMA model
def validate_verification_criteria(requirement_text):
   prompt = create_prompt(requirement_text)
  
   # Tokenize and move to device
   inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
 
   # Generate response
   with torch.no_grad():
       output = model.generate(inputs["input_ids"], max_length=500, do_sample=False)
  
   # Decode output
   decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
   return decoded_output.strip()
 
# Function to extract test components
def extract_test_details(requirement_text):
   preconditions = re.findall(r'Precondition: (.*)', requirement_text, re.IGNORECASE)
   inputs = re.findall(r'Input: (.*)', requirement_text, re.IGNORECASE)
   outputs = re.findall(r'Output: (.*)', requirement_text, re.IGNORECASE)
  
   return {
       "Preconditions": preconditions[0] if preconditions else "None (Suggested: " + criteria['preconditions'] + ")",
       "Input": inputs[0] if inputs else "Not specified (Suggested: Sensor data, User input, External trigger)",
       "Output": outputs[0] if outputs else "Not specified (Suggested: Log files, System response, Actuator movement)"
   }
 
# Validate and print results
def validate_text(requirement_text):
   print("\nProcessing Requirement:")
   print(requirement_text)
 
   test_details = extract_test_details(requirement_text)
   validation_result = validate_verification_criteria(requirement_text)
 
   print("\nTest Details:")
   print(f"- Preconditions: {test_details['Preconditions']}")
   print(f"- Input: {test_details['Input']}")
   print(f"- Output: {test_details['Output']}")
  
   print("\nValidation Results:")
   print(validation_result)
  
   return validation_result
 
# Dashboard function
def generate_dashboard(results):
   passed = sum(1 for result in results if "Yes" in result)
   failed = len(results) - passed
  
   labels = ['Passed', 'Failed']
   values = [passed, failed]
 
   plt.figure(figsize=(6, 4))
   plt.bar(labels, values, color=['green', 'red'])
   plt.xlabel("Test Case Status")
   plt.ylabel("Number of Test Cases")
   plt.title("Test Case Validation Results")
   plt.show()
 
# Main function to handle multiple inputs
def main():
   test_cases = [
       "Verification Method: Test\nInput:  Flash the Drive SW.\nOutput:  After the reset , periodic tasks should be running. \n",
       "Verification Method: Test\nAcceptance criteria: An active failure (RB_PDS_PDA_INTEGRITY_FAILED) stating Image verification failure shall be present and can be read through UDS service 0x19.\nPrecondition: Program BODA Image with manipulated user certificate.\n",
       "Verification Method: Test\nAcceptance criteria: An active failure (RB_PDS_PDA_AUTHENTICITY_FAILED) stating Image verification failure shall be present and can be read through UDS service 0x19.\nPrecondition: Manipulate B0 Digital signature of BODA block and program it into ECU.\n",
       "Verification Method: Test\nAcceptance criteria: An active failure stating Image verification failure shall be present and can be read through UDS service 0x19.\nPrecondition: Manipulate B3 data of FBL block and program it into ECU.\n"
   ]
  
   results = [validate_text(tc) for tc in test_cases]
  
   generate_dashboard(results)
 
if __name__ == "__main__":
   main()