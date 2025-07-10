import re
from climatefact.workflows.contradiction_detection.subgraphs.generation.types import GenerationState

def generate_report(state: GenerationState) -> GenerationState:
    """
    Generates a markdown report of the contradiction detection results.
    """
    print("---GENERATING REPORT---")
    
    contradiction_results = state.get("contradiction_results", [])
    
    # Calculate summary statistics
    total_sentences = len(contradiction_results)
    sentences_with_contradictions = sum(1 for result in contradiction_results if result["has_contradictions"])
    total_contradictions = sum(len(result["contradictions"]) for result in contradiction_results)
    
    report_lines = [
        "# 🔍 Contradiction Analysis Report",
        "",
        "## 📊 Summary",
        f"- **Total sentences analyzed:** {total_sentences}",
        f"- **Sentences with contradictions:** {sentences_with_contradictions}",
        f"- **Total contradictions found:** {total_contradictions}",
        ""
    ]
    
    if sentences_with_contradictions == 0:
        report_lines.extend([
            "## ✅ Results",
            "",
            "🎉 **No contradictions detected!** All analyzed sentences appear to be consistent with the available scientific evidence.",
            ""
        ])
    else:
        report_lines.extend([
            "## ⚠️ Contradictions Detected",
            "",
            f"The following {sentences_with_contradictions} sentence(s) contain contradictions with scientific evidence:",
            ""
        ])
        
        contradiction_count = 0
        for i, result in enumerate(contradiction_results):
            if result["has_contradictions"]:
                contradiction_count += 1
                sentence = result["sentence"]
                contradictions = result["contradictions"]
                
                report_lines.extend([
                    f"### Sentence {contradiction_count}",
                    "",
                    f"**📝 Analyzed Statement:**",
                    f"> {sentence}",
                    "",
                    f"**🚨 Found {len(contradictions)} contradiction(s):**",
                    ""
                ])
                
                for j, evidence in enumerate(contradictions, 1):
                    # Extract readable source information
                    source_name = evidence['source']['name']
                    actual_file_name = re.sub(r'_\d+\.md$', '', source_name)
                    page_number = evidence['source']['page']
                    
                    report_lines.extend([
                        f"#### Contradiction {j}",
                        "",
                        f"**📚 Contradictory Evidence:**",
                        f"> {evidence['contradictory_passage']}",
                        "",
                        f"**🔗 Source:** {actual_file_name} (Page {page_number})",
                        "",
                        "---",
                        ""
                    ])
    
    report_lines.extend([
        "",
        "**Note:** This analysis is based on available scientific literature and may not represent the complete body of evidence on climate topics. Always consult multiple authoritative sources for comprehensive understanding.",
        ""
    ])
            
    updated_state = state.copy()
    updated_state["report"] = "\n".join(report_lines)
    print("Report generated with enhanced formatting.")
    return updated_state
