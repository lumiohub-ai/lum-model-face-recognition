from gql import gql

LOGIN = gql('''
    mutation Login($input: LoginInput!) {
        login(input: $input) {
            _id
            memberType
            memberStatus
            memberAuthType
            memberPhone
            memberNick
            createdAt
            updatedAt
            accessToken
        }
    }
''')

RECORD_DATA = gql('''
    mutation CreateClientDate($input: DateInput!) {
        createClientDate(input: $input) {
            _id
            clientName
            clientIn
            clientOut
            clientWorkingHours
            clientWorkingDate
            clientId
        }
    }
''')