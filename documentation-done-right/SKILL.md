---
name: documentation-done-right
description: |
  Comprehensive guide for writing excellent technical documentation. Use when:
  (1) Creating docs for a new or existing codebase ("document this project", "create docs")
  (2) Updating documentation ("update the docs", "update docs for ../other-repo")
  (3) Auditing docs for accuracy ("verify docs", "check if docs are outdated", "find missing docs")
  (4) Creating or updating llms.txt/llms-full.txt files
  (5) Writing API documentation (endpoints, requests, responses)
  (6) Any documentation-related task including READMEs, guides, and technical references
  Triggers: "document", "documentation", "docs", "README", "llms.txt", "API docs", "outdated docs"
---

# Documentation Done Right

## Core Workflow

### Phase 1: Discovery

1. **Scan for existing documentation**
   ```
   Look for: README.md, docs/, documentation/, llms.txt, llms-full.txt,
   CONTRIBUTING.md, API.md, openapi.yaml, *.md files, JSDoc/docstrings
   ```

2. **Identify the documentation framework** (if any)
   - Check for: `mint.json` (Mintlify), `mkdocs.yml` (MkDocs), `docusaurus.config.js`, `.vitepress/`
   - See [references/frameworks.md](references/frameworks.md) for framework details

3. **Understand the codebase**
   - Entry points, main modules, API routes
   - Key abstractions and patterns
   - Dependencies and integrations

4. **Ask clarifying questions if unclear about:**
   - Target audience (developers, end-users, both?)
   - Scope (full docs, specific feature, API only?)
   - Existing documentation standards or templates
   - Whether to create new docs or update existing

### Phase 2: Audit (if docs exist)

Compare documentation against actual code:

1. **Check for outdated content**
   - Function signatures that changed
   - Removed features still documented
   - New features not documented
   - Changed configuration options
   - Updated dependencies

2. **Check for missing documentation**
   - Undocumented public APIs
   - Missing setup/installation steps
   - Undocumented environment variables
   - Missing error handling docs
   - No examples for complex features

3. **Check for inaccuracies**
   - Run documented commands to verify they work
   - Compare API examples against actual endpoints
   - Verify file paths and code references exist

4. **Report findings** before making changes:
   ```
   Found issues:
   - docs/api.md: POST /users endpoint removed in v2.0
   - README.md: Installation command uses deprecated flag
   - Missing: No docs for new /webhooks endpoints
   ```

### Phase 3: Write/Update Documentation

#### Style Guidelines

- **Be concise** - Developers skim; get to the point
- **Show, don't tell** - Examples over explanations
- **Use consistent formatting** - Match existing style
- **Include working examples** - Test code snippets
- **Document the "why"** - Not just the "what"

#### Structure for Different Doc Types

**README.md** (project root)
```markdown
# Project Name

One-line description.

## Quick Start
Fastest path to running the project.

## Installation
Step-by-step setup.

## Usage
Common use cases with examples.

## Configuration
Environment variables, options.

## API (brief)
Link to full API docs if extensive.

## Contributing
How to contribute.

## License
```

**API Documentation** - See [references/api-docs-guide.md](references/api-docs-guide.md)
- Document every endpoint with: method, path, description
- Request: headers, params, body with types and constraints
- Response: status codes, body schema, examples
- Errors: all possible error codes and meanings

**llms.txt / llms-full.txt** - See [references/llms-txt-spec.md](references/llms-txt-spec.md)
- `llms.txt`: Concise overview (~2000-4000 tokens)
- `llms-full.txt`: Comprehensive documentation
- Keep updated when codebase changes

### Phase 4: Verify

After writing/updating:

1. **Read through** for clarity and flow
2. **Test all code examples** - They must work
3. **Verify all links** - No broken references
4. **Check file paths** - All referenced files exist
5. **Ask user to review** if significant changes made

## When to Ask the User

Ask before proceeding when:

- **Scope is unclear**: "Should I document just the public API or internal modules too?"
- **Multiple valid approaches**: "Should I create a single README or a docs/ folder structure?"
- **Missing context**: "I see environment variables but no .env.example - what are the required vars?"
- **Significant decisions**: "The existing docs use Sphinx but MkDocs might be better for this. Preference?"
- **Uncertain about accuracy**: "The code shows 3 required params but docs say 2 - which is correct?"

## Common Tasks

### "Document this codebase"
1. Run discovery phase
2. Identify what exists vs what's needed
3. Ask about scope and audience
4. Create documentation structure
5. Write docs, starting with README and llms.txt
6. Add API docs if applicable

### "Update docs for ../other-repo"
1. Read the external repo's existing docs
2. Scan codebase for changes since last doc update
3. Identify discrepancies
4. Update docs to match current implementation
5. Verify accuracy

### "Find outdated docs"
1. Run full audit phase
2. Compare docs against code systematically
3. Report all discrepancies with specific locations
4. Offer to fix each issue

### "Create/update llms.txt"
1. Read existing llms.txt if present
2. Scan codebase for key information
3. Write concise llms.txt (overview, architecture, key files)
4. Write llms-full.txt if project is complex
5. See [references/llms-txt-spec.md](references/llms-txt-spec.md) for format

### "Write API documentation"
1. Find all API routes/endpoints
2. For each endpoint, document request and response
3. Include authentication requirements
4. Add working curl/code examples
5. Document all error responses
6. See [references/api-docs-guide.md](references/api-docs-guide.md) for patterns

## Quality Checklist

Before marking documentation complete:

- [ ] All public APIs documented
- [ ] Installation/setup instructions tested
- [ ] Code examples are copy-paste runnable
- [ ] No references to non-existent files/functions
- [ ] Environment variables documented
- [ ] Error scenarios covered
- [ ] llms.txt exists and is current (if appropriate for project)
- [ ] Consistent formatting throughout
- [ ] No TODO placeholders left
- [ ] User has reviewed significant changes
