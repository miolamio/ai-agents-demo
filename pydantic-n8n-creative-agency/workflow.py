"""
Content Creation Workflow with Pydantic AI
Multi-agent system for article creation with research, writing, review and editing
"""

import asyncio
import os
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import json

from openai import max_retries
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import httpx
from tavily import TavilyClient
from dotenv import load_dotenv

import logfire

# Шаг 0: Добавляем Logfire
logfire.configure(token="pylf_v1_us_DQN6NX83mbX9hcJz6yWWpRdgVFS47xxHJmKRzPhjfgsp") # Настраиваем Logfire
logfire.instrument_pydantic_ai() # Включаем автоматическую трассировку Pydantic!

# Load environment variables from .env file
load_dotenv()

# ==================== Configuration ====================
class Config:
    """Configuration for the workflow"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL = os.getenv("MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
    CONTEXT_WINDOW_LENGTH = int(os.getenv("CONTEXT_WINDOW_LENGTH", "20"))


# ==================== Data Models ====================
class ResearchResult(BaseModel):
    """Result from research phase"""
    topic: str
    summary: str = Field(description="Brief summary of the research findings")
    key_facts: List[str] = Field(description="List of key facts discovered")
    sources: List[str] = Field(description="List of source URLs")
    additional_info: Optional[str] = Field(default=None, description="Any additional relevant information")


class ArticleDraft(BaseModel):
    """Article draft structure"""
    title: str
    lead: str
    main_content: str
    conclusions: str
    metadata: Optional[Dict[str, Any]] = None


class ReviewResult(BaseModel):
    """Review feedback"""
    verdict: Literal["approved", "needs_revision"]
    critical_issues: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    reviewed_content: str


class FinalArticle(BaseModel):
    """Final edited article"""
    title: str
    lead: str
    content: str
    conclusions: str
    published_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Current state of the workflow"""
    stage: str
    research: Optional[ResearchResult] = None
    draft: Optional[ArticleDraft] = None
    review: Optional[ReviewResult] = None
    final_article: Optional[FinalArticle] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    iteration: int = 0


# ==================== Tools ====================
class TavilySearch:
    """Tavily search tool wrapper"""
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using Tavily"""
        try:
            # Tavily is synchronous, so we run it in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query,
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=True
                )
            )
            return response.get('results', [])
        except Exception as e:
            print(f"Search error: {e}")
            return []


# ==================== Agent Definitions ====================

# Set OpenAI API key as environment variable for the model
if Config.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

# Initialize model
model = OpenAIModel(Config.MODEL)

# Research Agent
research_agent = Agent(
    model=model,
    output_type=ResearchResult,
    retries=5,
    system_prompt="""You are a News Research Specialist, expert in searching and analyzing information.

YOUR ROLE:
Conduct detailed internet searches using Tavily to find the most current and reliable news. 
You provide the factual foundation for article creation.

YOUR GOALS:
1. Perform broad search queries to identify key news stories and trends
2. Verify all information found, relying only on verified and current sources
3. Don't make assumptions - all data must be fact-supported

PROCESS:
1. Analyze the request and identify key search topics
2. Use Tavily to search for current information
3. Verify source credibility
4. Systematize found information

RESULT:
Prepare a structured report with:
- Brief summary of the research findings
- List of key facts discovered
- List of source URLs
- Any additional relevant information""",
    deps_type=TavilySearch
)

@research_agent.tool
async def search_information(ctx: RunContext[TavilySearch], query: str) -> List[Dict[str, Any]]:
    """Search for information using Tavily"""
    return await ctx.deps.search(query)


# Writer Agent
writer_agent = Agent(
    model=model,
    output_type=ArticleDraft,
    system_prompt="""You are an Article Writer, professional content author.

YOUR ROLE:
Transform received research materials into a coherent, engaging article draft. 
The text should clearly and interestingly convey main facts to the audience.

YOUR GOALS:
1. Use research data to create a structured article
2. Present information correctly, based only on verified facts
3. Write the article according to editorial standards

ARTICLE STRUCTURE:
1. Title - attention-grabbing and essence-reflecting
2. Lead - brief presentation of main information
3. Main part - logically structured sections
4. Conclusions - summary of key points

WRITING STYLE:
- Clarity and accessibility
- Balance between informativeness and readability
- Use of active voice
- Short, clear sentences
- Logical transitions between paragraphs

RESULT:
Coherent article draft with main theses, facts, and conclusions, ready for review."""
)


# Reviewer Agent
reviewer_agent = Agent(
    model=model,
    output_type=ReviewResult,
    system_prompt="""You are an Article Reviewer. Your task is ONE-TIME draft review.

IMPORTANT: You have ONLY ONE chance to provide feedback. Be constructive and realistic.

REVIEW PRINCIPLES:
1. Focus ONLY on critical errors
2. Ignore minor stylistic issues (Editor will fix them)
3. Article doesn't have to be perfect
4. Better to approve a good article than demand perfection

CRITICAL ISSUES (require Writer's revision):
- Factual errors
- Missing key information
- Serious logical contradictions
- Complete topic mismatch

NON-CRITICAL ISSUES (pass to Editor):
- Stylistic improvements
- Minor inaccuracies
- Formatting
- Wording improvements

RESPONSE FORMAT:

Option 1 - Revisions needed:
"VERDICT: Needs revision
CRITICAL ISSUES (for Writer):
1. [only most important]
2. [only critical]

RECOMMENDATIONS FOR EDITOR:
- [stylistic improvements]"

Option 2 - Approved:
"VERDICT: Approved
Article ready for final editing.
RECOMMENDATIONS FOR EDITOR:
- [minor improvements if any]"

REMEMBER: You review ONLY ONCE. Don't demand perfection!"""
)


# Editor Agent
editor_agent = Agent(
    model=model,
    output_type=FinalArticle,
    system_prompt="""You are a Style Editor, professional copy editor.

YOUR ROLE:
Finally edit the article, bring it to uniform stylistic standards and prepare for publication.

YOUR GOALS:
1. Finalize the article and fix all errors
2. Ensure stylistic uniformity
3. Optimize text readability
4. Prepare material for publication

EDITING RULES:

Spelling and punctuation:
- Fix all spelling errors
- Place correct punctuation marks
- Use proper quotation marks
- Format dashes and hyphens correctly

Style:
- Remove tautologies and repetitions
- Simplify complex constructions
- Strengthen verbs, remove unnecessary adjectives
- Ensure terminology uniformity

Formatting:
- Format headings and subheadings
- Structure paragraphs (3-5 sentences)
- Format lists and enumerations
- Correctly write numbers, dates, names

Final check:
- Verify all facts and figures
- Ensure all sources are present
- Optimize headline for attention

RESULT:
Fully edited, publication-ready article in final form."""
)


# ==================== Supervisor (Orchestrator) ====================
class ContentCreationSupervisor:
    """Orchestrator for the content creation workflow"""
    
    def __init__(self):
        self.tavily = TavilySearch(Config.TAVILY_API_KEY)
        self.state = WorkflowState(stage="initial")
        self.model = model
    
    def log_stage(self, stage: str, message: str):
        """Log workflow stage"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "stage": stage,
            "message": message,
            "iteration": self.state.iteration
        }
        self.state.history.append(log_entry)
        print(f"[{timestamp}] {stage}: {message}")
    
    async def research_phase(self, topic: str) -> ResearchResult:
        """Execute research phase"""
        self.log_stage("RESEARCH", f"Starting research on topic: {topic}")
        
        # Run research agent with Tavily dependency
        result = await research_agent.run(
            f"Research the topic: {topic}. Find current information, facts, statistics about this topic.",
            deps=self.tavily
        )
        
        self.state.research = result.output
        self.state.stage = "research_complete"
        self.log_stage("RESEARCH", "Research completed")
        return result.output
    
    async def writing_phase(self, research: ResearchResult) -> ArticleDraft:
        """Execute writing phase"""
        self.log_stage("WRITING", "Starting article draft creation")
        
        prompt = f"""Based on the following research, create a structured article draft:

RESEARCH MATERIALS:
Topic: {research.topic}

Summary: {research.summary}

Key Facts:
{chr(10).join(f"• {fact}" for fact in research.key_facts)}

Sources:
{chr(10).join(f"• {source}" for source in research.sources)}

Additional Info: {research.additional_info or 'None available'}

Create an article with a title, lead, main body, and conclusions using the provided facts."""
        
        result = await writer_agent.run(prompt)
        self.state.draft = result.output
        self.state.stage = "draft_complete"
        self.log_stage("WRITING", "Draft completed")
        return result.output
    
    async def review_phase(self, draft: ArticleDraft) -> ReviewResult:
        """Execute review phase"""
        self.log_stage("REVIEW", "Starting article review")
        
        prompt = f"""Review the following article draft for accuracy and quality:

DRAFT FOR REVIEW:
Title: {draft.title}

Lead: {draft.lead}

Main Content:
{draft.main_content}

Conclusions:
{draft.conclusions}

Indicate ONLY critical issues. Give verdict: Approved or Needs revision."""
        
        result = await reviewer_agent.run(prompt)
        self.state.review = result.output
        self.state.stage = "review_complete"
        self.log_stage("REVIEW", f"Review completed - Verdict: {result.output.verdict}")
        return result.output
    
    async def revision_phase(self, draft: ArticleDraft, review: ReviewResult) -> ArticleDraft:
        """Execute revision phase if needed"""
        self.log_stage("REVISION", "Starting article revision")
        
        issues = chr(10).join(review.critical_issues) if review.critical_issues else ""
        prompt = f"""Fix critical issues in the article.

ORIGINAL ARTICLE:
Title: {draft.title}
Lead: {draft.lead}
Content: {draft.main_content}
Conclusions: {draft.conclusions}

REVIEWER'S CRITICAL ISSUES:
{issues}

Make ONLY the indicated fixes, don't rewrite the entire article."""
        
        result = await writer_agent.run(prompt)
        self.state.draft = result.output
        self.state.stage = "revision_complete"
        self.log_stage("REVISION", "Revision completed")
        return result.output
    
    async def editing_phase(self, draft: ArticleDraft, review: ReviewResult) -> FinalArticle:
        """Execute final editing phase"""
        self.log_stage("EDITING", "Starting final editing")
        
        recommendations = chr(10).join(review.recommendations) if review.recommendations else "None"
        prompt = f"""Perform final editing of the following article:

ARTICLE FOR EDITING:
Title: {draft.title}
Lead: {draft.lead}
Content: {draft.main_content}
Conclusions: {draft.conclusions}

REVIEWER'S RECOMMENDATIONS:
{recommendations}

Fix style, spelling, punctuation. Prepare for publication."""
        
        result = await editor_agent.run(prompt)
        result.output.published_at = datetime.now()
        self.state.final_article = result.output
        self.state.stage = "complete"
        self.log_stage("EDITING", "Final editing completed")
        return result.output
    
    async def run(self, topic: str) -> FinalArticle:
        """Run the complete workflow"""
        self.log_stage("WORKFLOW", f"Starting content creation workflow for: {topic}")
        
        try:
            # 1. Research Phase
            research = await self.research_phase(topic)
            
            # 2. Writing Phase
            draft = await self.writing_phase(research)
            
            # 3. Review Phase
            review = await self.review_phase(draft)
            
            # 4. Revision or Editing based on review
            if review.verdict == "needs_revision":
                if self.state.iteration >= Config.MAX_ITERATIONS:
                    self.log_stage("WORKFLOW", "Max iterations reached, proceeding to editing")
                    final = await self.editing_phase(draft, review)
                else:
                    self.state.iteration += 1
                    revised_draft = await self.revision_phase(draft, review)
                    # Re-review after revision
                    second_review = await self.review_phase(revised_draft)
                    final = await self.editing_phase(revised_draft, second_review)
            else:
                # 5. Direct to editing if approved
                final = await self.editing_phase(draft, review)
            
            self.log_stage("WORKFLOW", "Workflow completed successfully")
            return final
            
        except Exception as e:
            self.log_stage("ERROR", f"Workflow error: {str(e)}")
            raise
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary"""
        return {
            "current_stage": self.state.stage,
            "iterations": self.state.iteration,
            "history": self.state.history,
            "has_final_article": self.state.final_article is not None
        }


# ==================== Main Execution ====================
async def main():
    """Main execution function"""
    # Initialize supervisor
    supervisor = ContentCreationSupervisor()
    
    # Example topic
    topic = "Recent developments in AI and their impact on content creation"
    
    print("=" * 80)
    print("CONTENT CREATION WORKFLOW WITH PYDANTIC AI")
    print("=" * 80)
    
    # Run workflow
    final_article = await supervisor.run(topic)
    
    # Display results
    print("\n" + "=" * 80)
    print("FINAL ARTICLE")
    print("=" * 80)
    print(f"\nTitle: {final_article.title}")
    print(f"\nLead:\n{final_article.lead}")
    print(f"\nContent:\n{final_article.content}")
    print(f"\nConclusions:\n{final_article.conclusions}")
    
    # Display workflow summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    summary = supervisor.get_workflow_summary()
    print(json.dumps(summary, indent=2, default=str))


# ==================== Interactive Mode ====================
class InteractiveMode:
    """Interactive mode for the workflow"""
    
    def __init__(self):
        self.supervisor = ContentCreationSupervisor()
    
    async def run_interactive(self):
        """Run in interactive mode"""
        print("\n" + "=" * 80)
        print("INTERACTIVE CONTENT CREATION SYSTEM")
        print("=" * 80)
        print("\nCommands:")
        print("  /create <topic> - Create article on topic")
        print("  /status - Show current workflow status")
        print("  /history - Show workflow history")
        print("  /export - Export final article")
        print("  /quit - Exit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.startswith("/quit"):
                    print("Goodbye!")
                    break
                
                elif user_input.startswith("/create "):
                    topic = user_input[8:].strip()
                    if topic:
                        print(f"\nCreating article about: {topic}")
                        article = await self.supervisor.run(topic)
                        print(f"\n✅ Article created: {article.title}")
                    else:
                        print("Please provide a topic")
                
                elif user_input == "/status":
                    summary = self.supervisor.get_workflow_summary()
                    print(f"\nCurrent Stage: {summary['current_stage']}")
                    print(f"Iterations: {summary['iterations']}")
                    print(f"Has Final Article: {summary['has_final_article']}")
                
                elif user_input == "/history":
                    summary = self.supervisor.get_workflow_summary()
                    print("\nWorkflow History:")
                    for entry in summary['history']:
                        print(f"  [{entry['timestamp']}] {entry['stage']}: {entry['message']}")
                
                elif user_input == "/export":
                    if self.supervisor.state.final_article:
                        article = self.supervisor.state.final_article
                        
                        # Create articles directory if it doesn't exist
                        os.makedirs("articles", exist_ok=True)
                        
                        # Generate filename based on article title
                        safe_title = "".join(c for c in article.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        safe_title = safe_title.replace(' ', '_').lower()
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"articles/{safe_title}_{timestamp}.json"
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(article.model_dump(), f, indent=2, default=str, ensure_ascii=False)
                        print(f"✅ Article exported to {filename}")
                    else:
                        print("No article to export. Create one first with /create")
                
                else:
                    print("Unknown command. Type /quit to exit")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit")
            except Exception as e:
                print(f"Error: {e}")


# ==================== Entry Point ====================
if __name__ == "__main__":
    # Check for required environment variables
    if not Config.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found!")
        print("\n1. Create a .env file in the current directory")
        print("2. Add: OPENAI_API_KEY=your-key-here")
        print("\nOr set it as environment variable:")
        print("   export OPENAI_API_KEY=your-key-here")
        print("\nGet your API key at: https://platform.openai.com/api-keys")
        exit(1)
    
    if not Config.TAVILY_API_KEY:
        print("❌ TAVILY_API_KEY not found!")
        print("\n1. Create a .env file in the current directory")
        print("2. Add: TAVILY_API_KEY=your-key-here")
        print("\nOr set it as environment variable:")
        print("   export TAVILY_API_KEY=your-key-here")
        print("\nGet your free API key at: https://tavily.com/")
        exit(1)
    
    # Show loaded configuration
    print(f"✅ Configuration loaded:")
    print(f"   OpenAI Model: {Config.MODEL}")
    print(f"   OpenAI Key: {Config.OPENAI_API_KEY[:7]}...{Config.OPENAI_API_KEY[-4:]}")
    print(f"   Tavily Key: {Config.TAVILY_API_KEY[:4]}...{Config.TAVILY_API_KEY[-4:]}")
    
    # Run in interactive mode or single execution
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive = InteractiveMode()
        asyncio.run(interactive.run_interactive())
    else:
        # Single execution with example topic
        asyncio.run(main())