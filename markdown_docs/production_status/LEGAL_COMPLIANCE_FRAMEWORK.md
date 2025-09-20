---
title: Legal Compliance Framework Documentation
description: Auto-generated description for Legal Compliance Framework Documentation
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Legal Compliance Framework Documentation

## Overview

The JustNewsAgent project includes a comprehensive legal compliance framework designed to ensure full GDPR (General Data Protection Regulation) and CCPA (California Consumer Privacy Act) compliance. This framework provides enterprise-grade data protection, consent management, audit logging, and compliance monitoring capabilities.

## Architecture

### Core Components

#### 1. Data Minimization System
- **Purpose**: Ensures only necessary data is collected and processed
- **Implementation**: `agents/common/data_minimization.py`
- **Features**:
  - Automatic validation of data collection against minimization policies
  - 6 supported data purposes: contract fulfillment, legitimate interest, consent, marketing, profile analysis, data sharing
  - Real-time compliance monitoring

#### 2. Consent Management System
- **Purpose**: Manages user consent for data processing
- **Implementation**: `agents/common/consent_management.py`
- **Features**:
  - Granular consent tracking with expiration dates
  - Consent withdrawal capabilities
  - Audit logging for all consent operations
  - PostgreSQL database integration

#### 3. Audit Logging System
- **Purpose**: Maintains comprehensive audit trails for compliance
- **Implementation**: `agents/common/compliance_audit.py`
- **Features**:
  - GDPR article reference tracking
  - Tamper-proof audit entries
  - Structured logging with compliance event types
  - Real-time audit trail monitoring

#### 4. Consent Validation Middleware
- **Purpose**: Automatic consent validation for API endpoints
- **Implementation**: `agents/common/consent_validation_middleware.py`
- **Features**:
  - FastAPI middleware integration
  - Automatic consent checking before data processing
  - GDPR Article 6 compliance validation
  - Comprehensive error handling

#### 5. Compliance Dashboard
- **Purpose**: Real-time compliance monitoring and reporting
- **Implementation**: `agents/common/compliance_dashboard.py`
- **Features**:
  - Real-time compliance metrics
  - Audit trail visualization
  - Compliance violation alerts
  - Automated reporting

## API Endpoints

### Authentication Required Endpoints

All compliance endpoints require JWT authentication with appropriate roles:

- `ADMIN`: Full access to all compliance operations
- `RESEARCHER`: Read access to compliance data, limited write operations
- `VIEWER`: Read-only access to compliance reports

### Consent Management Endpoints

#### POST `/api/v1/compliance/consent`
Create or update user consent
```json
{
  "user_id": "string",
  "consent_type": "marketing|analytics|profiling|data_sharing|contract|legitimate_interest",
  "granted": true,
  "expires_at": "2025-12-31T23:59:59Z",
  "purpose": "string",
  "gdpr_article": "6|7|9"
}
```

#### GET `/api/v1/compliance/consent/{user_id}`
Retrieve user consent status
- Returns: Current consent status for all types

#### DELETE `/api/v1/compliance/consent/{user_id}`
Withdraw user consent (Right to be Forgotten)
- Removes all user data and consent records
- Maintains audit trail of deletion

#### GET `/api/v1/compliance/consent/{user_id}/export`
Export user data (Data Portability)
- Returns: User data in JSON, CSV, or XML format

### Audit Endpoints

#### GET `/api/v1/compliance/audit`
Retrieve audit logs
- Query parameters: `user_id`, `date_from`, `date_to`, `event_type`
- Returns: Paginated audit log entries

#### GET `/api/v1/compliance/audit/{entry_id}`
Retrieve specific audit entry
- Returns: Detailed audit entry with full context

### Compliance Monitoring Endpoints

#### GET `/api/v1/compliance/dashboard`
Compliance dashboard data
- Returns: Real-time compliance metrics and status

#### GET `/api/v1/compliance/reports`
Generate compliance reports
- Query parameters: `report_type`, `date_range`
- Returns: Compliance reports in various formats

## Data Subject Rights Implementation

### 1. Right to Access (GDPR Article 15)
- **Endpoint**: `GET /api/v1/compliance/consent/{user_id}/export`
- **Functionality**: Complete data export in multiple formats
- **Audit**: All access requests logged with GDPR Article 15 reference

### 2. Right to Rectification (GDPR Article 16)
- **Endpoint**: `PUT /api/v1/compliance/data/{user_id}`
- **Functionality**: Update inaccurate personal data
- **Audit**: All rectification requests logged

### 3. Right to Erasure (GDPR Article 17)
- **Endpoint**: `DELETE /api/v1/compliance/consent/{user_id}`
- **Functionality**: Complete data deletion and anonymization
- **Audit**: Deletion operations logged with data retention for legal compliance

### 4. Right to Data Portability (GDPR Article 20)
- **Endpoint**: `GET /api/v1/compliance/consent/{user_id}/export`
- **Functionality**: Data export in structured, machine-readable format
- **Formats**: JSON, CSV, XML

### 5. Right to Object (GDPR Article 21)
- **Endpoint**: `POST /api/v1/compliance/consent/{user_id}/object`
- **Functionality**: Object to processing based on legitimate interest
- **Audit**: Objection requests logged and processed

### 6. Right to Restriction (GDPR Article 18)
- **Endpoint**: `POST /api/v1/compliance/consent/{user_id}/restrict`
- **Functionality**: Restrict processing of personal data
- **Audit**: Restriction requests logged

## Lawful Basis for Processing

The system supports all GDPR lawful bases for data processing:

1. **Consent** (Article 6(1)(a)): Explicit user consent
2. **Contract** (Article 6(1)(b)): Processing necessary for contract performance
3. **Legal Obligation** (Article 6(1)(c)): Processing required by law
4. **Vital Interests** (Article 6(1)(d)): Processing to protect vital interests
5. **Public Task** (Article 6(1)(e)): Processing for public interest tasks
6. **Legitimate Interest** (Article 6(1)(f)): Processing for legitimate business interests

## Data Retention Policies

### Automatic Cleanup
- **Implementation**: Scheduled cleanup jobs
- **Retention Periods**: Configurable per data type
- **Audit**: All cleanup operations logged
- **Compliance**: GDPR Article 5(1)(e) compliance

### Data Categories
- **Personal Data**: 2 years default retention
- **Consent Records**: 7 years retention (legal requirement)
- **Audit Logs**: 10 years retention (compliance requirement)
- **Analytics Data**: 1 year retention

## Security Measures

### Database Security
- **Separate Databases**: User credentials and application data isolated
- **Encryption**: Data at rest and in transit encryption
- **Access Control**: Role-based access with principle of least privilege

### API Security
- **JWT Authentication**: Token-based authentication with expiration
- **Rate Limiting**: Protection against abuse
- **Input Validation**: Comprehensive input sanitization
- **Audit Logging**: All API operations logged

### Audit Trail Integrity
- **Tamper-Proof**: Cryptographic integrity validation
- **Immutable**: Audit entries cannot be modified after creation
- **Comprehensive**: All data operations logged with context

## UI Components

### Consent Banner
- **Location**: `agents/common/consent_ui_components.py`
- **Features**: GDPR-compliant consent banner with granular options
- **Responsive**: Mobile-friendly design
- **Accessibility**: WCAG 2.1 AA compliant

### Consent Modal
- **Features**: Detailed consent management interface
- **Granular Control**: Individual consent type management
- **Audit Integration**: All consent changes logged

### Compliance Dashboard
- **Real-time Monitoring**: Live compliance metrics
- **Audit Visualization**: Interactive audit trail viewer
- **Reporting**: Automated compliance reports

## Configuration

### Environment Variables
```bash
# Database Configuration
COMPLIANCE_DB_HOST=localhost
COMPLIANCE_DB_PORT=5432
COMPLIANCE_DB_NAME=justnews_compliance
COMPLIANCE_AUDIT_DB_NAME=justnews_audit

# Security Configuration
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Compliance Configuration
DATA_RETENTION_PERSONAL_DAYS=730
DATA_RETENTION_CONSENT_DAYS=2555
DATA_RETENTION_AUDIT_DAYS=3650
```

### Database Setup
The system requires two PostgreSQL databases:
1. **Main Compliance Database**: Stores consent and user data
2. **Audit Database**: Stores audit trails (separate for security)

## Monitoring and Alerting

### Compliance Metrics
- **Consent Coverage**: Percentage of users with valid consent
- **Audit Trail Health**: Audit logging system status
- **Data Minimization**: Compliance with minimization policies
- **Retention Compliance**: Data cleanup status

### Automated Alerts
- **Consent Expiration**: Alerts for expiring consent
- **Audit Failures**: Alerts for audit logging failures
- **Compliance Violations**: Alerts for policy violations
- **Data Retention**: Alerts for retention policy breaches

## Testing and Validation

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end compliance workflow testing
- **Compliance Tests**: GDPR requirement validation
- **Security Tests**: Penetration testing and vulnerability assessment

### Validation Scripts
- **Compliance Validator**: Automated GDPR compliance checking
- **Audit Validator**: Audit trail integrity validation
- **Data Flow Validator**: Data minimization policy validation

## Deployment Considerations

### Production Requirements
- **Database**: PostgreSQL 13+ with proper security configuration
- **Application Server**: FastAPI with Uvicorn
- **Security**: SSL/TLS encryption, proper firewall configuration
- **Monitoring**: Comprehensive logging and monitoring setup

### Scalability
- **Database Sharding**: Support for multiple database instances
- **Load Balancing**: Horizontal scaling capabilities
- **Caching**: Redis integration for performance optimization
- **Async Processing**: Background job processing for heavy operations

## Compliance Certification

### GDPR Compliance
- **Article 5**: Lawfulness, fairness, transparency, purpose limitation, data minimization, accuracy, storage limitation, integrity, confidentiality, accountability
- **Article 6**: Lawful basis for processing
- **Article 7**: Consent requirements
- **Articles 15-22**: Data subject rights
- **Article 25**: Data protection by design and default
- **Article 32**: Security of processing

### CCPA Compliance
- **Right to Know**: Access to personal information
- **Right to Delete**: Deletion of personal information
- **Right to Opt-Out**: Opt-out of sale of personal information
- **Right to Non-Discrimination**: Protection against discrimination

## Support and Maintenance

### Documentation Updates
- Regular updates to reflect regulatory changes
- Comprehensive API documentation
- User guide for compliance operations

### Training and Awareness
- Staff training on GDPR/CCPA requirements
- Regular compliance audits
- Incident response procedures

### Version Control
- All compliance-related code version controlled
- Change management for compliance features
- Audit trail of code changes

---

**Status**: **PRODUCTION READY**
**Last Updated**: 2025-09-02
**Version**: 1.0.0
**Compliance**: GDPR Article 5, 6, 7, 15-22, 25, 32 | CCPA Sections 1798.100, 1798.110, 1798.120, 1798.135

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

