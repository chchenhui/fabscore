"""
Data adapter to convert between different data formats
"""

def adapt_data_format(data_list):
    """
    Convert data from various formats to a consistent format
    Expected output format: {'text': str, 'label': str}
    """
    adapted_data = []

    for item in data_list:
        # Handle different input formats
        if isinstance(item, dict):
            # Extract text content
            text = ""
            if 'text' in item:
                text = item['text']
            elif 'body' in item:
                # Combine subject and body if available
                subject = item.get('subject', '')
                body = item.get('body', '')
                sender = item.get('sender', '')
                text = f"Subject: {subject}\n\nFrom: {sender}\n\n{body}"
            elif 'content' in item:
                text = item['content']
            elif 'message' in item:
                text = item['message']
            else:
                # Try to combine all string values
                text = ' '.join(str(v) for v in item.values() if isinstance(v, str))

            # Extract label
            label = None
            if 'label' in item:
                label_val = item['label']
                if isinstance(label_val, str):
                    label = label_val
                elif isinstance(label_val, int):
                    # Convert binary labels
                    label = 'phishing' if label_val == 1 else 'legitimate'
                elif isinstance(label_val, bool):
                    label = 'phishing' if label_val else 'legitimate'
            elif 'is_phishing' in item:
                label = 'phishing' if item['is_phishing'] else 'legitimate'
            elif 'class' in item:
                label = item['class']
            elif 'category' in item:
                label = item['category']
            else:
                # Default to legitimate if no label found
                label = 'legitimate'

            adapted_data.append({
                'text': text,
                'label': label
            })

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Handle tuple/list format (text, label)
            text = str(item[0])
            label_val = item[1]

            if isinstance(label_val, str):
                label = label_val
            elif isinstance(label_val, int):
                label = 'phishing' if label_val == 1 else 'legitimate'
            else:
                label = 'legitimate'

            adapted_data.append({
                'text': text,
                'label': label
            })

        elif isinstance(item, str):
            # Plain text, assume legitimate
            adapted_data.append({
                'text': item,
                'label': 'legitimate'
            })

    return adapted_data