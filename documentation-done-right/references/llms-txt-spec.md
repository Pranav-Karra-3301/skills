# llms.txt Specification

Reference: https://llmstxt.org/

## Purpose

llms.txt provides a standardized way to give LLMs concise, relevant information about a project or website. It lives at `/llms.txt` (or in the repo root) and helps AI assistants understand and work with codebases effectively.

## File Types

### llms.txt (Concise)
- Brief overview of the project
- Key entry points and architecture
- Most important files/endpoints
- Target: ~2000-4000 tokens

### llms-full.txt (Comprehensive)
- Complete documentation
- All API endpoints with examples
- Full configuration options
- Detailed architecture explanations
- Target: As comprehensive as needed

## Standard Format

```markdown
# Project Name

> Brief one-line description of what this project does.

## Overview

2-3 sentences explaining the project's purpose, target users, and core value proposition.

## Quick Start

Essential commands to get started:
- Installation
- Basic usage
- Key entry point

## Architecture

Brief description of:
- Main components/modules
- How they interact
- Key design decisions

## Key Files

- `src/index.ts` - Main entry point
- `src/api/` - API route handlers
- `src/lib/` - Core utilities
- `src/types/` - TypeScript definitions

## API Overview (if applicable)

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/users | GET | List users |
| /api/users/:id | GET | Get user by ID |

## Configuration

Key environment variables or config options.

## Common Tasks

How to accomplish frequent operations.
```

## Best Practices

1. **Be concise in llms.txt** - LLMs work better with focused context
2. **Use llms-full.txt for depth** - Put detailed examples, full API docs here
3. **Keep updated** - Outdated llms.txt is worse than none
4. **Link to detailed docs** - Reference full documentation for deep dives
5. **Include examples** - Show, don't just tell
