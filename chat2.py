import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

def test_api_key() -> bool:
    """Test if the API key is valid and working."""
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not found in .env file")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content("Test")
        
        return True
    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have created a .env file in the project directory")
        print("2. The .env file contains: GOOGLE_API_KEY=your_actual_api_key")
        print("3. You have installed required packages: pip install python-dotenv google-generativeai")
        print("4. Your API key is valid and has access to Gemini Pro")
        return False

class WebsiteRequirementsChatbot:
    """
    A dynamic conversational AI chatbot for gathering website requirements,
    using contextual understanding to generate relevant questions.
    """

    def __init__(self, output_dir: str = "website_requirements_output"):
        """Initialize the chatbot with error handling."""
        try:
            if not self._setup_api():
                raise SystemExit(1)
            
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            self.model = genai.GenerativeModel("gemini-pro")
            self.conversation_history: List[Dict[str, str]] = []
            self.asked_questions = set()
            self.question_count = 0
            self.min_questions = 15
            self.final_brief_generated = False
            self.retry_attempts = 3
            self.user_expertise = None
            self.website_type = None
            
            self.system_prompt = self._get_system_prompt()
            
        except Exception as e:
            print(f"\n❌ Initialization failed: {str(e)}")
            raise SystemExit(1)

    def _setup_api(self) -> bool:
        """Configure Gemini API with error handling."""
        try:
            if not os.path.exists('.env'):
                raise FileNotFoundError("'.env' file not found")
            
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
                
            genai.configure(api_key=api_key)
            return True
            
        except Exception as e:
            print(f"\n❌ API Setup Error: {str(e)}")
            return False

    def _get_system_prompt(self) -> str:
        """Return the dynamic system prompt that guides the AI's questioning strategy."""
        return """
        You are a specialized website requirements gathering assistant. Your role is to conduct an 
        intelligent conversation to understand and document website requirements.

        CONVERSATION STRATEGY:
        1. First Interaction:
           - Understand the type of website needed (e-commerce, portfolio, business, blog, etc.)
           - Identify the primary goals and purpose

        2. Dynamic Questioning:
           - Based on the website type and previous answers, generate relevant follow-up questions
           - Each question should build upon previous responses
           - Adapt the technical level based on user expertise
           - Ask ONE question at a time
           - Ensure questions cover all crucial aspects of website development

        3. Key Areas to Cover (adapt based on website type):
           - Core functionality and features
           - User experience and interface preferences
           - Content management needs
           - Visual design preferences
           - Technical requirements
           - Security and performance needs
           - Future scalability considerations
           - Budget and timeline constraints
           - Maintenance and update requirements

        4. Question Guidelines:
           - Keep questions clear and concise
           - Provide relevant examples when needed
           - Avoid technical jargon unless user shows expertise
           - Focus on business goals rather than technical implementation
           - Ensure questions flow logically
           - Adapt to user's responses and level of detail
           - if user not answered correctly, don't ask same question or relevant question again
           - make sure after 15 questios. you generate final prompt.

        5. Final Brief Format:
        After gathering sufficient information, create a comprehensive brief using this structure:

        BEGIN_BRIEF
        1. Project Overview
           - Website type and purpose
           - Primary goals and objectives
           - Target audience

        2. Functional Requirements
           - Core features
           - User interactions
           - Content management needs

        3. Design Requirements
           - Visual style preferences
           - Brand integration
           - User interface priorities

        4. Technical Specifications
           - Platform requirements
           - Integration needs
           - Security requirements

        5. Content Strategy
           - Content types
           - Update frequency
           - Management approach

        6. Additional Considerations
           - Scalability needs
           - Performance requirements
           
           

        
        END_BRIEF

        RULES:
        1. Ask questions in order of priority for the specific website type , ask around 15 questions only.
        2. Maintain context between questions
        3. If user seems unclear, provide examples or clarification and  Leave the question and you put related answer to that question in the final prompt.
        4. Generate the final brief only after gathering comprehensive information
        5. Make sure questions and brief are tailored to the specific website type
        6. Ensure all critical aspects are covered before generating the brief
        """

    def _assess_expertise(self, response: str) -> str:
        """Determine user's technical expertise level."""
        technical_terms = ["html", "css", "javascript", "api", "database", "hosting", "dns", "ssl"]
        response_lower = response.lower()
        
        if any(term in response_lower for term in technical_terms):
            return "advanced"
        elif "don't know" in response_lower or "not sure" in response_lower:
            return "beginner"
        return "intermediate"

    def format_conversation(self) -> str:
        """Format the conversation history for the AI model."""
        formatted = self.system_prompt + "\n\n"
        for entry in self.conversation_history:
            formatted += f"{entry['role']}: {entry['content']}\n"
        return formatted + "\nBased on previous responses, ask your next relevant question."

    def get_next_question(self, user_response: Optional[str] = None) -> str:
        """Get the next contextual question from the AI."""
        if user_response:
            if self.user_expertise is None:
                self.user_expertise = self._assess_expertise(user_response)
            self.conversation_history.append({"role": "User", "content": user_response})

        self.question_count += 1

        for attempt in range(self.retry_attempts):
            try:
                prompt = self.format_conversation()
                response = self.model.generate_content(prompt)
                
                if hasattr(response, 'text'):
                    question = response.text.strip()
                elif hasattr(response, 'parts'):
                    question = ''.join(part.text for part in response.parts).strip()
                else:
                    raise ValueError("Invalid response format")

                if not question or question in self.asked_questions:
                    raise ValueError("Invalid or duplicate question")

                self.asked_questions.add(question)
                self.conversation_history.append({"role": "Assistant", "content": question})
                return question

            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    print(f"\n⚠️ Question generation failed: {str(e)}")
                    return "Could you tell me more about your website needs?"

        return "What other requirements do you have for your website?"

    def save_brief(self, content: str) -> Tuple[str, str]:
        """Save the website requirements brief in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        md_file = f"{self.output_dir}/website_brief_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(content)
            
        html_file = f"{self.output_dir}/website_brief_{timestamp}.html"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Website Requirements Brief</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                    line-height: 1.6;
                }}
                h1, h2 {{ color: #2c3e50; }}
                h2 {{ margin-top: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
                .content {{ background: #f9f9f9; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Website Requirements Brief</h1>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <div class="content">
                {content.replace('\n', '<br>')}
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
            
        return md_file, html_file

    def save_state(self) -> None:
        """Save the current conversation state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/conversation_state_{timestamp}.json"
        
        state = {
            "conversation_history": self.conversation_history,
            "question_count": self.question_count,
            "asked_questions": list(self.asked_questions),
            "final_brief_generated": self.final_brief_generated,
            "user_expertise": self.user_expertise,
            "website_type": self.website_type
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

    def run(self) -> None:
        """Run the chatbot interaction."""
        print("\n🤖 Website Planning Assistant")
        print("I'll help you plan your website - let's start with understanding your needs!")
        print("\nCommands:")
        print("- Type 'quit' to end the session")
        print("- Type 'save' to save progress")
        print("- Type 'help' for suggestions")
        print("- Type 'not sure' if you're unsure about any question\n")

        try:
            first_question = "What kind of website are you looking to build? Tell me about your goals and what you want to achieve with this website."
            print(f"Assistant: {first_question}")

            while not self.final_brief_generated:
                user_input = input("\nYour response: ").strip()
                command = user_input.lower()
                
                if command == 'quit':
                    print("\n👋 Saving and ending session...")
                    self.save_state()
                    break
                    
                if command == 'save':
                    self.save_state()
                    print("\n💾 Progress saved!")
                    continue

                if command == 'help':
                    print("\n💡 Tips:")
                    print("- Share your website's main purpose")
                    print("- Think about your target audience")
                    print("- Consider must-have features")
                    print("- Don't worry about technical details yet")
                    continue

                if not user_input:
                    print("Please provide a response or type 'help' for suggestions.")
                    continue

                question = self.get_next_question(user_input)

                if self.question_count >= self.min_questions and "BEGIN_BRIEF" in question:
                    self.final_brief_generated = True
                    print("\n📋 Creating Your Website Plan...")
                    print("=" * 80)
                    print(question)
                    print("=" * 80)

                    md_file, html_file = self.save_brief(question)
                    print(f"\n💾 Your website plan has been saved:")
                    print(f"- Readable version: {html_file}")
                    print(f"- Text version: {md_file}")
                else:
                    print(f"\nAssistant: {question}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Session interrupted. Saving progress...")
            self.save_state()
            print("Progress saved!")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Saving progress...")
            self.save_state()
            print("Progress saved!")

def main():
    """Main function to run the chatbot application."""
    print("\n🌟 Website Requirements Gathering Assistant")
    print("==========================================")
    print("\n📋 Initializing system...")
    
    if test_api_key():
        try:
            print("\n✅ API key verified")
            print("Starting the requirements gathering process...\n")
            chatbot = WebsiteRequirementsChatbot()
            chatbot.run()
        except Exception as e:
            print(f"\n❌ Error running chatbot: {str(e)}")
            print("Please check your setup and try again.")
    else:
        print("\n❌ System initialization failed.")
        print("Please check your API key and setup, then try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
    finally:
        print("\n💫 Thank you for using the Website Requirements Assistant!")