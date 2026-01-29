# API Documentation Patterns

## Request Documentation

Every API endpoint needs:

```markdown
### POST /api/resource

Create a new resource.

**Request Headers**
| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token |
| Content-Type | Yes | application/json |

**Request Body**
```json
{
  "name": "string (required) - Resource name, 1-100 chars",
  "type": "string (optional) - One of: 'A', 'B', 'C'. Default: 'A'",
  "metadata": {
    "key": "value (optional) - Additional data"
  }
}
```

**Example Request**
```bash
curl -X POST https://api.example.com/api/resource \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Resource", "type": "B"}'
```
```

## Response Documentation

```markdown
**Response** `201 Created`
```json
{
  "id": "res_abc123",
  "name": "My Resource",
  "type": "B",
  "createdAt": "2024-01-15T10:30:00Z"
}
```

**Error Responses**

| Status | Code | Description |
|--------|------|-------------|
| 400 | INVALID_NAME | Name exceeds 100 characters |
| 401 | UNAUTHORIZED | Missing or invalid token |
| 409 | DUPLICATE | Resource with name already exists |

**Error Response Body**
```json
{
  "error": {
    "code": "INVALID_NAME",
    "message": "Name must be between 1 and 100 characters"
  }
}
```
```

## Documentation Checklist

### For Each Endpoint
- [ ] HTTP method and path
- [ ] Brief description of what it does
- [ ] Authentication requirements
- [ ] Request headers (with required/optional)
- [ ] Path parameters
- [ ] Query parameters (with defaults)
- [ ] Request body schema (with types and constraints)
- [ ] Example request (curl or language-specific)
- [ ] Success response (status code + body)
- [ ] Error responses (all possible errors)
- [ ] Rate limiting info (if applicable)

### For the API Overall
- [ ] Base URL
- [ ] Authentication method (how to get tokens)
- [ ] Rate limiting policy
- [ ] Pagination pattern
- [ ] Error response format
- [ ] Versioning strategy
- [ ] SDKs/clients available

## Common Patterns

### Pagination
```markdown
**Query Parameters**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| page | integer | 1 | Page number |
| limit | integer | 20 | Items per page (max 100) |

**Response includes:**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "hasMore": true
  }
}
```
```

### Filtering
```markdown
**Query Parameters**
| Param | Type | Description |
|-------|------|-------------|
| status | string | Filter by status: active, inactive, pending |
| createdAfter | ISO8601 | Filter by creation date |
| search | string | Full-text search on name/description |

Example: `GET /api/users?status=active&createdAfter=2024-01-01`
```

### Webhooks
```markdown
## Webhook Events

### user.created
Fired when a new user signs up.

**Payload**
```json
{
  "event": "user.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "id": "usr_abc123",
    "email": "user@example.com"
  }
}
```

**Signature Verification**
```python
import hmac
signature = hmac.new(secret, payload, 'sha256').hexdigest()
assert signature == request.headers['X-Webhook-Signature']
```
```
