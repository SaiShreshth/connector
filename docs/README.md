# ChatNShop Documentation Hub

Welcome to the comprehensive documentation for the ChatNShop Intent Classification System!

## 📚 Documentation Index

### Getting Started (Start Here!)

1. **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** ⭐ **START HERE**
   - Installation steps
   - Running the system
   - Testing examples
   - Troubleshooting common issues
   - **Read this first to get the system running!**

2. **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)**
   - High-level system overview
   - Component relationships
   - Technology stack and rationale
   - Performance characteristics
   - Deployment architecture

3. **[DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)**
   - Complete file tree
   - Purpose of each directory
   - Purpose of each key file
   - File interaction diagrams

### Testing and Quality

4. **[TESTING_AND_COVERAGE.md](TESTING_AND_COVERAGE.md)**
   - Testing strategy and test organization
   - Coverage reports and quality metrics
   - Why each approach was chosen
   - Code examples and algorithms
   - **Most comprehensive document - 16,000+ words**

5. **[SYSTEM_FLOW.md](../SYSTEM_FLOW.md)** (Root directory)
   - End-to-end request flows
   - Decision engine logic
   - Orchestration details
   - Visual flow diagrams

### Subsystem Documentation

#### Intent Classification (Rule-Based)
Located in `docs/intent_classification/`:

- **[ARCHITECTURE.md](intent_classification/ARCHITECTURE.md)** - Component architecture
- **[API.md](intent_classification/API.md)** - API documentation
- **[KEYWORD_MAINTENANCE.md](intent_classification/KEYWORD_MAINTENANCE.md)** - How to maintain keywords
- **[KEYWORD_MATCHING_RATIONALE.md](intent_classification/KEYWORD_MATCHING_RATIONALE.md)** - Why keyword approach
- **[EMBEDDINGS.md](intent_classification/EMBEDDINGS.md)** - Embedding system details
- **[PERFORMANCE_TUNING.md](intent_classification/PERFORMANCE_TUNING.md)** - Optimization guide
- **[TROUBLESHOOTING.md](intent_classification/TROUBLESHOOTING.md)** - Common issues
- **[DECISION_FLOW.md](intent_classification/DECISION_FLOW.md)** - Decision logic
- **[CONFIG_EXAMPLES.md](intent_classification/CONFIG_EXAMPLES.md)** - Configuration samples
- **[CONFIG_VERSIONING.md](intent_classification/CONFIG_VERSIONING.md)** - Version control
- **[INTENT_TAXONOMY_APPROVAL.md](intent_classification/INTENT_TAXONOMY_APPROVAL.md)** - Intent taxonomy
- **[INTENT_MAPPING_EXAMPLES.md](intent_classification/INTENT_MAPPING_EXAMPLES.md)** - Query → Intent examples
- **[EXAMPLES_INDEX.md](intent_classification/EXAMPLES_INDEX.md)** - Example queries

#### LLM Intent Classification
Located in `docs/llm_intent/`:

- **[CACHING.md](llm_intent/CACHING.md)** - LLM caching strategies

#### Entity Extraction
Located in `docs/entity_extraction/`:

- **[ENTITY_EXTRACTION.md](entity_extraction/ENTITY_EXTRACTION.md)** - Entity extraction system

#### Error Handling
Located in `docs/error_handling/`:

- **[ERROR_HANDLING.md](error_handling/ERROR_HANDLING.md)** - Error handling strategy

#### Cost Monitoring
Located in `docs/cost_monitoring/`:

- **[COST_MONITORING.md](cost_monitoring/COST_MONITORING.md)** - Cost tracking system
- **[PROMPT_OPTIMIZATION.md](cost_monitoring/PROMPT_OPTIMIZATION.md)** - Prompt optimization

#### Testing Framework
Located in `docs/testing_framework/`:

- **[TESTING_FRAMEWORK.md](testing_framework/TESTING_FRAMEWORK.md)** - A/B testing framework

### Additional Resources

6. **[TESTING_AND_COVERAGE.md](TESTING_AND_COVERAGE.md)**
   - Test structure
   - Running tests
   - Coverage reports


---

## 🎯 Quick Navigation by Role

### For New Developers

**Day 1: Getting Started**
1. Read [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Install and run
2. Read [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Understand the big picture
3. Read [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) - Know where everything is

**Day 2: Deep Dive**
4. Read [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Deep architecture understanding
5. Read [intent_classification/ARCHITECTURE.md](intent_classification/ARCHITECTURE.md) - Rule-based system
6. Browse code with new understanding

**Day 3: Hands-On**
7. Make small changes (add keywords)
8. Run tests
9. Read [TESTING_AND_COVERAGE.md](TESTING_AND_COVERAGE.md)

### For DevOps/SRE

**Focus On**:
1. [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Deployment
2. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Infrastructure section
3. Docker configuration (docker-compose.yml)
4. [intent_classification/TROUBLESHOOTING.md](intent_classification/TROUBLESHOOTING.md)
5. Health check endpoints
6. Monitoring and alerts

### For Product Managers

**Focus On**:
1. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - System capabilities
2. [intent_classification/INTENT_TAXONOMY_APPROVAL.md](intent_classification/INTENT_TAXONOMY_APPROVAL.md) - All intents
3. [intent_classification/INTENT_MAPPING_EXAMPLES.md](intent_classification/INTENT_MAPPING_EXAMPLES.md) - Examples
4. [cost_monitoring/COST_MONITORING.md](cost_monitoring/COST_MONITORING.md) - Costs

### For ML Engineers

**Focus On**:
1. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - LLM and AI components
2. [intent_classification/EMBEDDINGS.md](intent_classification/EMBEDDINGS.md) - Embedding system
3. [cost_monitoring/PROMPT_OPTIMIZATION.md](cost_monitoring/PROMPT_OPTIMIZATION.md) - Prompt engineering
4. [testing_framework/TESTING_FRAMEWORK.md](testing_framework/TESTING_FRAMEWORK.md) - A/B testing

### For QA/Testers

**Focus On**:
1. [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Running and testing
2. [TESTING_AND_COVERAGE.md](TESTING_AND_COVERAGE.md) - Test structure
3. [intent_classification/EXAMPLES_INDEX.md](intent_classification/EXAMPLES_INDEX.md) - Test cases
4. Test files in `tests/` directory

---

## 📊 Documentation Statistics

- **Python Version**: 3.12+ (recommended for best performance)
- **Total Documentation Files**: 30+
- **Total Words**: 50,000+
- **Total Code Examples**: 200+
- **Total Diagrams**: 15+

### Core Documents
- PROJECT_ARCHITECTURE.md: 7,500 words
- DIRECTORY_STRUCTURE.md: 8,000 words
- QUICK_START_GUIDE.md: 4,500 words
- TESTING_AND_COVERAGE.md: 5,000 words

---

## 🔍 Finding What You Need

### By Topic

**Understanding the System**:
- Architecture → [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
- File structure → [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)
- Data flows → [SYSTEM_FLOW.md](../SYSTEM_FLOW.md)

**Getting It Running**:
- Quick start → [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
- Deployment → [QUICK_START_GUIDE.md#running-the-system](QUICK_START_GUIDE.md#running-the-system)
- Docker → docker-compose.yml + Dockerfile

**Development**:
- Architecture → [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
- Adding keywords → [intent_classification/KEYWORD_MAINTENANCE.md](intent_classification/KEYWORD_MAINTENANCE.md)
- Testing → [TESTING_AND_COVERAGE.md](TESTING_AND_COVERAGE.md)

**Troubleshooting**:
- Common issues → [QUICK_START_GUIDE.md#troubleshooting](QUICK_START_GUIDE.md#troubleshooting)
- Intent classification → [intent_classification/TROUBLESHOOTING.md](intent_classification/TROUBLESHOOTING.md)
- Error handling → [error_handling/ERROR_HANDLING.md](error_handling/ERROR_HANDLING.md)

### By Feature Area

All features and components are explained in detail in [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md):

**Rule-Based System**:
- Task 1: Intent taxonomy
- Task 2: Keyword dictionaries
- Task 3: Keyword matching
- Task 4: Embedding matching
- Task 5: Hybrid classifier
- Task 6: Ambiguity handling
- Task 7: API implementation
- Task 8: Testing
- Task 9: Configuration
- Task 10: Documentation

**LLM (Tasks 11-23)**:
- Task 11: LLM requirements
- Task 12: Prompt engineering
- Task 13: GPT-4 integration
- Task 14: Queue infrastructure
- Task 15: Queue producer
- Task 16: Status tracking
- Task 17: Context enhancement
- Task 18: Confidence calibration
- Task 19: Caching
- Task 20: Entity extraction
- Task 21: Error handling
- Task 22: Cost monitoring
- Task 23: A/B testing

---

## 🚀 Recommended Reading Order

### For Complete Understanding (4-6 hours)

1. **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** (30 min)
   - Get system running
   - Test basic queries
   - Understand flows

2. **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)** (45 min)
   - System overview
   - Component relationships
   - Design decisions

3. **[DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)** (45 min)
   - File organization
   - Module purposes
   - Dependencies

4. **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)** (2-3 hours)
   - Complete system architecture
   - Implementation specifics
   - Code examples

5. **Subsystem Docs** (1 hour)
   - Pick relevant sections
   - Deep dive on areas of interest

### For Quick Start (1 hour)

1. [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) (30 min)
2. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Skim (15 min)
3. [intent_classification/EXAMPLES_INDEX.md](intent_classification/EXAMPLES_INDEX.md) (15 min)

### For Specific Tasks

**Adding a new intent**:
1. [intent_classification/INTENT_TAXONOMY_APPROVAL.md](intent_classification/INTENT_TAXONOMY_APPROVAL.md)
2. [intent_classification/KEYWORD_MAINTENANCE.md](intent_classification/KEYWORD_MAINTENANCE.md)
3. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Intent system architecture

**Tuning performance**:
1. [intent_classification/PERFORMANCE_TUNING.md](intent_classification/PERFORMANCE_TUNING.md)
2. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Performance section
3. [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Configuration tuning

**Debugging issues**:
1. [QUICK_START_GUIDE.md#troubleshooting](QUICK_START_GUIDE.md#troubleshooting)
2. [intent_classification/TROUBLESHOOTING.md](intent_classification/TROUBLESHOOTING.md)
3. [error_handling/ERROR_HANDLING.md](error_handling/ERROR_HANDLING.md)

---

## 💡 Tips for Using This Documentation

1. **Start with QUICK_START_GUIDE** - Get hands-on experience first
2. **Use Ctrl+F / Cmd+F** - Search within documents
3. **Follow links** - Documents are heavily cross-referenced
4. **Check examples** - Every concept has code examples
5. **Run tests** - Validate your understanding

---

## 🔄 Keeping Documentation Updated

This documentation is version-controlled alongside code. When making changes:

1. **Update relevant docs** when changing functionality
2. **Add examples** for new features
3. **Update PROJECT_ARCHITECTURE** for major changes
4. **Keep version history** in changelog files

---

## 📞 Getting Help

1. **Check docs first** - Most questions answered here
2. **Search issues** - Someone may have asked before
3. **Check logs** - `logs/ambiguous.jsonl` for classification issues
4. **Run tests** - `pytest tests/` to isolate problems
5. **Check health** - `curl http://localhost:8000/health`

---

## 📈 Documentation Roadmap

### Completed ✅
- System architecture
- All 23 tasks explained
- Directory structure
- Quick start guide
- Subsystem documentation
- Troubleshooting guides

### Future Additions 🔮
- Video tutorials
- Interactive examples
- Performance benchmarking guide
- Migration guides
- Contribution guidelines

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Maintained By**: ChatNShop Development Team

---

## Quick Links

- 🏠 [Project Root](../)
- 🚀 [Quick Start](QUICK_START_GUIDE.md)
- 🏗️ [Architecture](PROJECT_ARCHITECTURE.md)
- 📁 [Directory Structure](DIRECTORY_STRUCTURE.md)
- 🧪 [Testing](TESTING_AND_COVERAGE.md)
- 💰 [Cost Monitoring](cost_monitoring/COST_MONITORING.md)
- 🔧 [Troubleshooting](QUICK_START_GUIDE.md#troubleshooting)

