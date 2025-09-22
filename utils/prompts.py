"""
Prompt Templates for AI Analytics Agents

This module provides structured prompt templates for the Analyser and Report agents
with fraud detection domain knowledge, JSON schema expectations, and different
analysis type templates.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
from dataclasses import dataclass
from datetime import datetime


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    DATA_DESCRIPTION = "data_description"
    PATTERN_IDENTIFICATION = "pattern_identification"
    OUTLIER_DETECTION = "outlier_detection"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    FRAUD_ANALYSIS = "fraud_analysis"
    COMPREHENSIVE = "comprehensive"


class AgentRole(Enum):
    """AI agent roles."""
    ANALYSER = "analyser"
    REPORTER = "reporter"


@dataclass
class PromptTemplate:
    """Template structure for prompts."""
    role: AgentRole
    analysis_type: AnalysisType
    system_prompt: str
    user_prompt_template: str
    expected_output_schema: Dict[str, Any]
    domain_knowledge: str
    examples: List[Dict[str, Any]]


class FraudDetectionPrompts:
    """
    Fraud detection domain-specific prompt templates for AI agents.

    Provides structured prompts for analyzing financial transaction data with
    PCA-transformed features (V1-V28), Time, Amount, and Class variables.
    """

    # Fraud detection domain knowledge
    FRAUD_DOMAIN_KNOWLEDGE = """
    ## Fraud Detection Domain Knowledge

    ### Dataset Context:
    - **PCA Features (V1-V28)**: Principal components from PCA transformation of original features
    - **Time**: Seconds elapsed between each transaction and the first transaction
    - **Amount**: Transaction amount in currency units
    - **Class**: Binary fraud indicator (0=normal, 1=fraud)

    ### Key Fraud Patterns:
    1. **Temporal Patterns**: Fraudulent transactions often occur at unusual hours or in bursts
    2. **Amount Patterns**: Fraud may involve very small amounts (testing) or very large amounts (maximizing gain)
    3. **Feature Anomalies**: PCA components may show unusual patterns for fraudulent transactions
    4. **Imbalanced Nature**: Fraud is typically <1% of all transactions (highly imbalanced dataset)

    ### Statistical Considerations:
    - Use appropriate tests for imbalanced data
    - Consider both statistical significance and practical significance
    - Outliers may be fraud cases or data quality issues
    - Correlation patterns may reveal fraud detection features

    ### Analysis Goals:
    - Identify distinguishing characteristics of fraudulent vs normal transactions
    - Detect temporal and amount-based fraud patterns
    - Evaluate feature importance for fraud detection
    - Assess data quality and preprocessing needs
    """

    # JSON Schema for Analyser Agent output
    ANALYSER_OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "analysis_summary": {
                "type": "object",
                "properties": {
                    "dataset_overview": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "fraud_insights": {"type": "string"},
                    "data_quality_assessment": {"type": "string"}
                },
                "required": ["dataset_overview", "key_findings", "fraud_insights"]
            },
            "statistical_analysis": {
                "type": "object",
                "properties": {
                    "descriptive_statistics": {"type": "object"},
                    "distribution_analysis": {"type": "object"},
                    "correlation_findings": {"type": "object"},
                    "outlier_analysis": {"type": "object"}
                }
            },
            "fraud_specific_analysis": {
                "type": "object",
                "properties": {
                    "fraud_rate": {"type": "number"},
                    "temporal_patterns": {"type": "object"},
                    "amount_patterns": {"type": "object"},
                    "feature_importance": {"type": "object"},
                    "anomaly_detection": {"type": "object"}
                }
            },
            "recommendations": {
                "type": "object",
                "properties": {
                    "data_preprocessing": {"type": "array", "items": {"type": "string"}},
                    "feature_engineering": {"type": "array", "items": {"type": "string"}},
                    "model_recommendations": {"type": "array", "items": {"type": "string"}},
                    "further_analysis": {"type": "array", "items": {"type": "string"}}
                }
            },
            "confidence_scores": {
                "type": "object",
                "properties": {
                    "analysis_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "data_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "pattern_strength": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        },
        "required": ["analysis_summary", "fraud_specific_analysis", "recommendations"]
    }

    # JSON Schema for Report Agent output
    REPORTER_OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "report_metadata": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "generated_at": {"type": "string"},
                    "analysis_type": {"type": "string"},
                    "executive_summary": {"type": "string"}
                },
                "required": ["title", "analysis_type", "executive_summary"]
            },
            "markdown_report": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_title": {"type": "string"},
                        "content": {"type": "string"},
                        "charts": {"type": "array", "items": {"type": "string"}},
                        "tables": {"type": "array"}
                    }
                }
            },
            "key_metrics": {
                "type": "object",
                "properties": {
                    "fraud_rate": {"type": "number"},
                    "total_transactions": {"type": "integer"},
                    "data_quality_score": {"type": "number"},
                    "analysis_confidence": {"type": "number"}
                }
            }
        },
        "required": ["report_metadata", "markdown_report", "key_metrics"]
    }

    def __init__(self):
        """Initialize the fraud detection prompt templates."""
        self.templates = self._create_all_templates()

    def _create_all_templates(self) -> Dict[str, PromptTemplate]:
        """Create all prompt templates."""
        templates = {}

        # Analyser Agent templates
        templates.update(self._create_analyser_templates())

        # Reporter Agent templates
        templates.update(self._create_reporter_templates())

        return templates

    def _create_analyser_templates(self) -> Dict[str, PromptTemplate]:
        """Create Analyser Agent prompt templates."""
        templates = {}

        # Comprehensive Analysis Template
        templates["analyser_comprehensive"] = PromptTemplate(
            role=AgentRole.ANALYSER,
            analysis_type=AnalysisType.COMPREHENSIVE,
            system_prompt=f"""You are an expert data analyst specializing in fraud detection and financial transaction analysis. Your task is to analyze statistical summaries and pattern analysis results from fraud detection datasets.

{self.FRAUD_DOMAIN_KNOWLEDGE}

## Your Responsibilities:
1. Analyze provided statistical data and pattern analysis results
2. Identify key insights about fraud vs normal transaction patterns
3. Evaluate data quality and preprocessing needs
4. Provide actionable recommendations for fraud detection systems
5. Output results in the specified JSON format with confidence scores

## Analysis Approach:
- Focus on practical insights that can improve fraud detection
- Consider statistical significance AND business impact
- Highlight unusual patterns that may indicate fraud
- Assess data quality issues that could affect model performance
- Provide clear, actionable recommendations

Be thorough but concise. Your analysis will be used to generate reports for both technical and business stakeholders.""",

            user_prompt_template="""Please analyze the following fraud detection dataset statistics and patterns:

## Dataset Statistics:
{statistics_data}

## Pattern Analysis Results:
{pattern_data}

## Visualization Metadata:
{chart_metadata}

## Analysis Instructions:
Perform a comprehensive analysis focusing on:

1. **Dataset Overview**: Summarize the key characteristics of this fraud detection dataset
2. **Fraud Patterns**: Identify patterns that distinguish fraudulent from normal transactions
3. **Statistical Insights**: Highlight significant statistical findings
4. **Data Quality**: Assess data quality and potential issues
5. **Actionable Recommendations**: Provide specific recommendations for fraud detection improvement

Please provide your analysis in valid JSON format following the expected schema. Include confidence scores for your findings.""",

            expected_output_schema=self.ANALYSER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[
                {
                    "input_type": "comprehensive_analysis",
                    "expected_output": {
                        "analysis_summary": {
                            "dataset_overview": "Credit card fraud dataset with 284,807 transactions, 0.17% fraud rate, 28 PCA features plus Time and Amount",
                            "key_findings": [
                                "Highly imbalanced dataset requiring specialized handling",
                                "Significant temporal patterns in fraud occurrence",
                                "Amount distributions differ between fraud and normal transactions"
                            ],
                            "fraud_insights": "Fraudulent transactions show distinct temporal clustering and amount patterns, suggesting systematic fraud behavior",
                            "data_quality_assessment": "High quality dataset with no missing values, PCA transformation preserves privacy while maintaining analytical value"
                        }
                    }
                }
            ]
        )

        # Data Description Template
        templates["analyser_data_description"] = PromptTemplate(
            role=AgentRole.ANALYSER,
            analysis_type=AnalysisType.DATA_DESCRIPTION,
            system_prompt=f"""You are a data analyst specializing in descriptive statistics for fraud detection datasets. Focus on characterizing the data types, distributions, and basic statistical properties.

{self.FRAUD_DOMAIN_KNOWLEDGE}

Provide detailed analysis of data characteristics, paying special attention to fraud detection context.""",

            user_prompt_template="""Analyze the data description aspects of this fraud detection dataset:

Statistics: {statistics_data}
Pattern Data: {pattern_data}

Focus on:
- Data types and their appropriateness for fraud detection
- Distribution characteristics of each variable
- Range analysis and central tendency measures
- Variability assessment
- Missing values and data quality

Provide analysis in JSON format.""",

            expected_output_schema=self.ANALYSER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        # Pattern Identification Template
        templates["analyser_pattern_identification"] = PromptTemplate(
            role=AgentRole.ANALYSER,
            analysis_type=AnalysisType.PATTERN_IDENTIFICATION,
            system_prompt=f"""You are a pattern recognition expert for fraud detection systems. Focus on identifying temporal patterns, trends, clusters, and frequency patterns that may indicate fraudulent behavior.

{self.FRAUD_DOMAIN_KNOWLEDGE}

Look for patterns that distinguish fraud from normal transactions.""",

            user_prompt_template="""Identify patterns in this fraud detection dataset:

Statistics: {statistics_data}
Pattern Analysis: {pattern_data}

Focus on:
- Temporal patterns and trends
- Frequency analysis of different values
- Clustering patterns in the data
- Seasonal or periodic behaviors
- Fraud-specific pattern identification

Provide detailed pattern analysis in JSON format.""",

            expected_output_schema=self.ANALYSER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        # Outlier Detection Template
        templates["analyser_outlier_detection"] = PromptTemplate(
            role=AgentRole.ANALYSER,
            analysis_type=AnalysisType.OUTLIER_DETECTION,
            system_prompt=f"""You are an outlier detection specialist for fraud detection systems. Analyze outliers and their potential relationship to fraudulent transactions.

{self.FRAUD_DOMAIN_KNOWLEDGE}

Consider that outliers may be fraud cases, data quality issues, or legitimate edge cases.""",

            user_prompt_template="""Analyze outliers in this fraud detection dataset:

Statistics: {statistics_data}
Pattern Analysis: {pattern_data}

Focus on:
- Outlier identification across different variables
- Relationship between outliers and fraud labels
- Impact of outliers on analysis
- Recommendations for outlier treatment
- Data cleaning vs fraud detection considerations

Provide outlier analysis in JSON format.""",

            expected_output_schema=self.ANALYSER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        # Relationship Analysis Template
        templates["analyser_relationship_analysis"] = PromptTemplate(
            role=AgentRole.ANALYSER,
            analysis_type=AnalysisType.RELATIONSHIP_ANALYSIS,
            system_prompt=f"""You are a correlation and relationship analysis expert for fraud detection datasets. Focus on variable relationships and their implications for fraud detection.

{self.FRAUD_DOMAIN_KNOWLEDGE}

Analyze how variables relate to each other and to the fraud target variable.""",

            user_prompt_template="""Analyze variable relationships in this fraud detection dataset:

Statistics: {statistics_data}
Pattern Analysis: {pattern_data}

Focus on:
- Variable correlations and relationships
- Feature importance for fraud detection
- Cross-variable influence patterns
- Interaction effects
- Multicollinearity considerations

Provide relationship analysis in JSON format.""",

            expected_output_schema=self.ANALYSER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        return templates

    def _create_reporter_templates(self) -> Dict[str, PromptTemplate]:
        """Create Reporter Agent prompt templates."""
        templates = {}

        # Comprehensive Report Template
        templates["reporter_comprehensive"] = PromptTemplate(
            role=AgentRole.REPORTER,
            analysis_type=AnalysisType.COMPREHENSIVE,
            system_prompt=f"""You are a professional data analyst and report writer specializing in fraud detection and financial crime analytics. Your task is to create comprehensive, well-structured markdown reports from analysis results.

{self.FRAUD_DOMAIN_KNOWLEDGE}

## Report Requirements:
1. Create clear, professional markdown reports suitable for both technical and business audiences
2. Include executive summary with key insights and recommendations
3. Structure content logically with proper headings and sections
4. Incorporate visualizations and tables effectively
5. Highlight fraud-specific insights and actionable recommendations
6. Ensure reports are complete, accurate, and actionable

## Report Structure Guidelines:
- Executive Summary (2-3 paragraphs)
- Dataset Overview
- Key Findings (bullet points)
- Detailed Analysis Sections
- Fraud Detection Insights
- Recommendations
- Technical Details (appendix)

Make reports engaging and informative while maintaining professional standards.""",

            user_prompt_template="""Create a comprehensive fraud detection analysis report based on the following analysis results:

## Analysis Results:
{analysis_results}

## Available Visualizations:
{available_charts}

## Dataset Context:
- Original file: {dataset_name}
- Analysis timestamp: {timestamp}
- Analysis type: {analysis_type}

## Report Requirements:
Please create a professional markdown report that includes:

1. **Executive Summary**: High-level insights and key findings
2. **Dataset Overview**: Basic characteristics and data quality assessment
3. **Fraud Analysis**: Specific fraud detection insights and patterns
4. **Statistical Findings**: Key statistical results and their implications
5. **Visualizations**: Reference and describe relevant charts/graphs
6. **Recommendations**: Actionable next steps for fraud detection improvement
7. **Technical Details**: Methodology and confidence assessments

The report should be suitable for presentation to both technical teams and business stakeholders. Include specific metrics, percentages, and quantitative insights where available.

Provide the output in the specified JSON format with the complete markdown report.""",

            expected_output_schema=self.REPORTER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[
                {
                    "input_type": "comprehensive_report",
                    "expected_output": {
                        "report_metadata": {
                            "title": "Fraud Detection Analysis Report",
                            "analysis_type": "comprehensive",
                            "executive_summary": "Analysis of 284,807 credit card transactions reveals a 0.17% fraud rate with distinct temporal and amount patterns that can improve detection systems."
                        },
                        "markdown_report": "# Fraud Detection Analysis Report\n\n## Executive Summary\n...",
                        "key_metrics": {
                            "fraud_rate": 0.0017,
                            "total_transactions": 284807,
                            "data_quality_score": 0.95,
                            "analysis_confidence": 0.88
                        }
                    }
                }
            ]
        )

        # Technical Report Template
        templates["reporter_technical"] = PromptTemplate(
            role=AgentRole.REPORTER,
            analysis_type=AnalysisType.COMPREHENSIVE,
            system_prompt=f"""You are a technical analyst creating detailed reports for data scientists and machine learning engineers working on fraud detection systems.

{self.FRAUD_DOMAIN_KNOWLEDGE}

Focus on technical details, statistical methodologies, and implementation recommendations.""",

            user_prompt_template="""Create a technical fraud detection analysis report:

Analysis Results: {analysis_results}
Charts: {available_charts}
Dataset: {dataset_name}

Include:
- Technical methodology
- Statistical significance tests
- Feature engineering recommendations
- Model development insights
- Code examples and technical details

Format as detailed markdown report in JSON structure.""",

            expected_output_schema=self.REPORTER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        # Executive Summary Template
        templates["reporter_executive"] = PromptTemplate(
            role=AgentRole.REPORTER,
            analysis_type=AnalysisType.COMPREHENSIVE,
            system_prompt=f"""You are a business analyst creating executive summaries for senior management and business stakeholders interested in fraud detection insights.

{self.FRAUD_DOMAIN_KNOWLEDGE}

Focus on business impact, risk assessment, and strategic recommendations.""",

            user_prompt_template="""Create an executive summary report for fraud detection analysis:

Analysis Results: {analysis_results}
Key Metrics: {key_metrics}
Business Context: {business_context}

Include:
- Business impact assessment
- Risk quantification
- Strategic recommendations
- Resource requirements
- ROI considerations

Format as concise executive markdown report in JSON structure.""",

            expected_output_schema=self.REPORTER_OUTPUT_SCHEMA,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        return templates

    def get_template(self, agent_role: AgentRole, analysis_type: AnalysisType) -> PromptTemplate:
        """
        Get a specific prompt template.

        Args:
            agent_role: The agent role (ANALYSER or REPORTER)
            analysis_type: The type of analysis

        Returns:
            PromptTemplate object

        Raises:
            KeyError: If template not found
        """
        template_key = f"{agent_role.value}_{analysis_type.value}"

        if template_key not in self.templates:
            # Fallback to comprehensive template
            fallback_key = f"{agent_role.value}_comprehensive"
            if fallback_key in self.templates:
                return self.templates[fallback_key]
            else:
                raise KeyError(f"Template not found: {template_key}")

        return self.templates[template_key]

    def format_prompt(self,
                     agent_role: AgentRole,
                     analysis_type_enum: AnalysisType,
                     **kwargs) -> Dict[str, str]:
        """
        Format a prompt template with provided data.

        Args:
            agent_role: The agent role
            analysis_type_enum: The type of analysis
            **kwargs: Data to fill in the template

        Returns:
            Dictionary with system_prompt and user_prompt
        """
        template = self.get_template(agent_role, analysis_type_enum)

        # Format the user prompt template with provided data
        try:
            formatted_user_prompt = template.user_prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template parameter: {e}")

        return {
            "system_prompt": template.system_prompt,
            "user_prompt": formatted_user_prompt,
            "expected_schema": template.expected_output_schema,
            "domain_knowledge": template.domain_knowledge
        }

    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available templates.

        Returns:
            Dictionary with template information
        """
        template_info = {}

        for template_key, template in self.templates.items():
            template_info[template_key] = {
                "role": template.role.value,
                "analysis_type": template.analysis_type.value,
                "description": f"{template.role.value.title()} template for {template.analysis_type.value.replace('_', ' ')} analysis",
                "required_parameters": self._extract_template_parameters(template.user_prompt_template)
            }

        return template_info

    def _extract_template_parameters(self, template_string: str) -> List[str]:
        """Extract parameter names from template string."""
        import re
        return re.findall(r'\{(\w+)\}', template_string)

    def create_custom_template(self,
                              agent_role: AgentRole,
                              analysis_type: AnalysisType,
                              system_prompt: str,
                              user_prompt_template: str,
                              expected_output_schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a custom template.

        Args:
            agent_role: Agent role
            analysis_type: Analysis type
            system_prompt: System prompt
            user_prompt_template: User prompt template
            expected_output_schema: Expected output schema

        Returns:
            Template key for the created template
        """
        template_key = f"{agent_role.value}_{analysis_type.value}_custom"

        if expected_output_schema is None:
            expected_output_schema = (self.ANALYSER_OUTPUT_SCHEMA if agent_role == AgentRole.ANALYSER
                                    else self.REPORTER_OUTPUT_SCHEMA)

        custom_template = PromptTemplate(
            role=agent_role,
            analysis_type=analysis_type,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            expected_output_schema=expected_output_schema,
            domain_knowledge=self.FRAUD_DOMAIN_KNOWLEDGE,
            examples=[]
        )

        self.templates[template_key] = custom_template
        return template_key

    def validate_output_schema(self, output_data: Dict[str, Any], agent_role: AgentRole) -> bool:
        """
        Validate output against expected schema.

        Args:
            output_data: Data to validate
            agent_role: Agent role to determine schema

        Returns:
            True if valid, False otherwise
        """
        try:
            expected_schema = (self.ANALYSER_OUTPUT_SCHEMA if agent_role == AgentRole.ANALYSER
                             else self.REPORTER_OUTPUT_SCHEMA)

            # Basic schema validation (simplified)
            if not isinstance(output_data, dict):
                return False

            required_fields = expected_schema.get("required", [])
            for field in required_fields:
                if field not in output_data:
                    return False

            return True
        except Exception:
            return False


# Convenience functions
def get_fraud_prompts() -> FraudDetectionPrompts:
    """Get fraud detection prompt templates instance."""
    return FraudDetectionPrompts()


def format_analyser_prompt(analysis_type: AnalysisType,
                          statistics_data: str,
                          pattern_data: str,
                          chart_metadata: str = "") -> Dict[str, str]:
    """
    Quick function to format analyser prompt.

    Args:
        analysis_type: Type of analysis
        statistics_data: JSON string of statistics
        pattern_data: JSON string of pattern analysis
        chart_metadata: Optional chart metadata

    Returns:
        Formatted prompt dictionary
    """
    prompts = get_fraud_prompts()
    return prompts.format_prompt(
        agent_role=AgentRole.ANALYSER,
        analysis_type_enum=analysis_type,
        statistics_data=statistics_data,
        pattern_data=pattern_data,
        chart_metadata=chart_metadata
    )


def format_reporter_prompt(analysis_results: str,
                          available_charts: str = "",
                          dataset_name: str = "fraud_dataset",
                          timestamp: str = None,
                          analysis_type_name: str = "comprehensive") -> Dict[str, str]:
    """
    Quick function to format reporter prompt.

    Args:
        analysis_results: JSON string of analysis results
        available_charts: Description of available charts
        dataset_name: Name of the dataset
        timestamp: Analysis timestamp
        analysis_type_name: Type of analysis

    Returns:
        Formatted prompt dictionary
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    prompts = get_fraud_prompts()
    return prompts.format_prompt(
        agent_role=AgentRole.REPORTER,
        analysis_type_enum=AnalysisType.COMPREHENSIVE,
        analysis_results=analysis_results,
        available_charts=available_charts,
        dataset_name=dataset_name,
        timestamp=timestamp,
        analysis_type=analysis_type_name
    )


if __name__ == "__main__":
    # Example usage and testing
    print("=== Fraud Detection Prompt Templates ===")

    prompts = get_fraud_prompts()

    # Show available templates
    templates = prompts.get_available_templates()
    print(f"Available templates: {len(templates)}")

    for template_key, info in templates.items():
        print(f"  - {template_key}: {info['description']}")
        print(f"    Required parameters: {info['required_parameters']}")

    # Test template formatting
    print("\n=== Template Testing ===")

    try:
        sample_prompt = format_analyser_prompt(
            AnalysisType.COMPREHENSIVE,
            statistics_data='{"test": "data"}',
            pattern_data='{"test": "patterns"}',
            chart_metadata='{"charts": ["histogram", "correlation"]}'
        )

        print("✓ Analyser prompt formatted successfully")
        print(f"System prompt length: {len(sample_prompt['system_prompt'])} characters")
        print(f"User prompt length: {len(sample_prompt['user_prompt'])} characters")

    except Exception as e:
        print(f"✗ Error formatting analyser prompt: {e}")

    try:
        sample_report_prompt = format_reporter_prompt(
            analysis_results='{"test": "analysis"}',
            available_charts="histogram, correlation heatmap",
            dataset_name="test_dataset.csv"
        )

        print("✓ Reporter prompt formatted successfully")
        print(f"System prompt length: {len(sample_report_prompt['system_prompt'])} characters")

    except Exception as e:
        print(f"✗ Error formatting reporter prompt: {e}")

    print("\n✅ Prompt templates ready for agent integration!")