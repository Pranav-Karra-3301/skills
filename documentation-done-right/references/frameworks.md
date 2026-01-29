# Documentation Frameworks

## Static Site Generators

### Mintlify
- **Best for**: API documentation, developer docs
- **URL**: https://mintlify.com/docs
- **Config**: `mint.json`
- **Features**: API playground, search, analytics, versioning

### MkDocs (Material theme)
- **Best for**: Technical documentation, Python projects
- **URL**: https://squidfunk.github.io/mkdocs-material/
- **Config**: `mkdocs.yml`
- **Features**: Search, dark mode, code annotations

### Docusaurus
- **Best for**: Open source projects, versioned docs
- **URL**: https://docusaurus.io/
- **Config**: `docusaurus.config.js`
- **Features**: Versioning, i18n, blog, MDX support

### VitePress
- **Best for**: Vue projects, lightweight docs
- **URL**: https://vitepress.dev/
- **Config**: `.vitepress/config.ts`
- **Features**: Fast builds, Vue components in markdown

### Nextra
- **Best for**: Next.js projects
- **URL**: https://nextra.site/
- **Config**: `next.config.js` + `theme.config.tsx`
- **Features**: MDX, full-text search, i18n

### GitBook
- **Best for**: Team documentation, knowledge bases
- **URL**: https://www.gitbook.com/
- **Features**: WYSIWYG editor, Git sync, integrations

## API Documentation

### OpenAPI/Swagger
- **Spec**: https://swagger.io/specification/
- **Tools**: Swagger UI, Redoc, Stoplight
- **Files**: `openapi.yaml` or `openapi.json`

### Redoc
- **Best for**: Beautiful OpenAPI rendering
- **URL**: https://redocly.com/redoc/

### Stoplight
- **Best for**: API design-first workflow
- **URL**: https://stoplight.io/

## Code Documentation

### JSDoc (JavaScript/TypeScript)
- **URL**: https://jsdoc.app/
- **Generate HTML**: `jsdoc -c jsdoc.json`

### TypeDoc (TypeScript)
- **URL**: https://typedoc.org/
- **Config**: `typedoc.json`

### Sphinx (Python)
- **URL**: https://www.sphinx-doc.org/
- **Config**: `conf.py`
- **Popular theme**: Read the Docs

### rustdoc (Rust)
- **Built-in**: `cargo doc`

### GoDoc (Go)
- **URL**: https://pkg.go.dev/

## Choosing a Framework

| Project Type | Recommended |
|--------------|-------------|
| API-first product | Mintlify, Redoc |
| Open source library | Docusaurus, MkDocs |
| Python project | Sphinx, MkDocs |
| JavaScript/TypeScript | TypeDoc + Docusaurus |
| Internal docs | GitBook, Notion |
| Simple project | Plain Markdown READMEs |
